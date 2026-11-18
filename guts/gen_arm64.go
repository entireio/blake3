//go:build ignore

// gen_arm64 generates compress_arm64.s, the four-lane BLAKE3 kernels for
// arm64. Run it from this directory:
//
//	go run gen_arm64.go -out compress_arm64.s
//
// Pass -stats to print how many instructions of each kind one block of four
// lanes costs.
//
// # Layout
//
// Every kernel keeps the sixteen state words of four independent compressions
// in V0-V15, one word per register and one lane per 32-bit element, so a G
// function is a handful of whole-register operations and a round needs no
// shuffles. The rounds live in a subroutine per instruction set that the
// kernels call with BL; it expects the transposed message words wherever
// (variant).msg says they are, and preserves everything but the state and
// its scratch registers.
//
//	NEON: V16-V26 message words 0-10, V27 rotate-by-8 table, V28-V31 scratch,
//	      message words 11-15 in the caller's frame.
//	SVE2: V16-V31 message words 0-15.
//
// NEON has no rotate instruction. Rotating by 16 is a halfword swap and
// rotating by 8 a byte shuffle; the other two XOR into a scratch register and
// then shift left and shift-right-insert into the destination, which is three
// instructions for "xor, rotate" instead of four. Four scratch registers let
// the four G functions of a half round stay interleaved, which matters to
// cores that do not rename registers.
//
// SVE2 has XAR, "xor and rotate right", which collapses each of those pairs
// into one instruction and needs no scratch register or table. The Go
// assembler does not know SVE, so XAR is emitted as a WORD with its mnemonic
// alongside. Only the low 128 bits of each Z register are meaningful here:
// NEON instructions zero the rest and XAR treats every element independently,
// so the kernels are correct for any vector length.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
)

var iv = [4]uint32{0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A}

// msgPermutation is applied to the message words between rounds.
var msgPermutation = [16]int{2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}

// Offsets into the constant table.
const (
	constIV       = 0   // four IV words, each replicated across the lanes
	constBlockLen = 64  // 64 in every lane
	constRot8     = 80  // byte shuffle that rotates each word right by 8
	constLanes01  = 96  // 64-bit lane indices 0, 1
	constLanes23  = 112 // 64-bit lane indices 2, 3
	constSize     = 128
)

// Offsets from RSP into the kernels' frames. 0(RSP) holds the saved link
// register, so nothing may be stored below 8(RSP); starting at 16 keeps every
// slot addressable with a scaled load.
const (
	frameMsg       = 16  // message words that do not fit in registers
	frameCounterLo = 96  // low halves of the lane counters
	frameCounterHi = 112 // high halves of the lane counters
	frameSize      = 136 // locals occupy 8(RSP) up to 144(RSP)
)

// A variant describes one instruction set's register plan.
type variant struct {
	name    string
	sve2    bool
	msgRegs int // message words kept in registers, starting at V16
}

var (
	neon = variant{name: "NEON", msgRegs: 11}
	sve2 = variant{name: "SVE2", sve2: true, msgRegs: 16}
)

const (
	rot8Table = 27 // NEON only
	scratch   = 28 // NEON only: V28-V31
)

// msgReg reports the register holding message word i, if any.
func (v variant) msgReg(i int) (int, bool) {
	if i < v.msgRegs {
		return 16 + i, true
	}
	return 0, false
}

// msgSlot is the frame offset of message word i when it has no register.
func (v variant) msgSlot(i int) int {
	return frameMsg + 16*(i-v.msgRegs)
}

type gen struct {
	buf     bytes.Buffer
	section string
	counts  map[string]map[string]int
}

// count attributes the instruction to the current section, by kind.
func (g *gen) count(ins string) {
	op, args, _ := strings.Cut(ins, " ")
	var kind string
	switch {
	case op == "WORD":
		kind = "vector"
	case strings.HasPrefix(op, "VLD1"):
		kind = "load"
	case op == "FMOVQ" || op == "MOVD" || op == "MOVWU":
		switch {
		case strings.HasPrefix(args, "F") && strings.Contains(args, "("):
			kind = "store"
		case strings.Contains(args, "(") && !strings.HasPrefix(args, "$"):
			kind = "load"
		default:
			kind = "integer"
		}
	case op == "VDUP" || op == "VMOV":
		kind = "integer" // general-purpose to vector transfer
	case strings.HasPrefix(op, "V"):
		kind = "vector"
	default:
		kind = "integer"
	}
	if g.counts[g.section] == nil {
		g.counts[g.section] = map[string]int{}
	}
	g.counts[g.section][kind]++
}

func (g *gen) ins(format string, args ...any) {
	s := fmt.Sprintf(format, args...)
	g.count(s)
	fmt.Fprintf(&g.buf, "\t%s\n", s)
}

func (g *gen) raw(format string, args ...any) {
	fmt.Fprintf(&g.buf, format+"\n", args...)
}

func (g *gen) comment(format string, args ...any) {
	fmt.Fprintf(&g.buf, "\t// "+format+"\n", args...)
}

func (g *gen) label(name string) {
	fmt.Fprintf(&g.buf, "%s:\n", name)
}

// xar emits Zdn = ror(Zdn ^ Zm, n) on 32-bit elements.
func (g *gen) xar(dn, m, n int) {
	// 00000100 tszh 1 tszl imm3 001101 Zm Zdn, with tszh:tszl:imm3 = 64-n
	// for 32-bit elements.
	enc := 64 - uint32(n)
	word := 0x04203400 | enc>>5<<22 | enc>>3&3<<19 | enc&7<<16 | uint32(m)<<5 | uint32(dn)
	g.ins("WORD $0x%08x // xar z%d.s, z%d.s, z%d.s, #%d", word, dn, dn, m, n)
}

func (g *gen) consts() {
	g.raw("// IV words replicated across the lanes, the block length, the byte shuffle")
	g.raw("// that rotates each word right by 8, and the 64-bit lane indices.")
	data := func(off int, words ...uint32) {
		for i, w := range words {
			g.raw("DATA blake3consts<>+%d(SB)/4, $0x%08x", off+4*i, w)
		}
	}
	for i, w := range iv {
		data(constIV+16*i, w, w, w, w)
	}
	data(constBlockLen, 64, 64, 64, 64)
	data(constRot8, 0x00030201, 0x04070605, 0x080b0a09, 0x0c0f0e0d)
	data(constLanes01, 0, 0, 1, 0)
	data(constLanes23, 2, 0, 3, 0)
	g.raw("GLOBL blake3consts<>(SB), RODATA|NOPTR, $%d", constSize)
}

// addMsg adds one message word to each of the four registers in a.
func (g *gen) addMsg(v variant, a [4]int, words [4]int) {
	for i, w := range words {
		if r, ok := v.msgReg(w); ok {
			g.ins("VADD V%d.S4, V%d.S4, V%d.S4", r, a[i], a[i])
		} else {
			g.ins("FMOVQ %d(RSP), F%d", v.msgSlot(w), scratch+i)
			g.ins("VADD V%d.S4, V%d.S4, V%d.S4", scratch+i, a[i], a[i])
		}
	}
}

// each emits one instruction per G function of a half round.
func (g *gen) each(format string, regs ...[4]int) {
	for i := 0; i < 4; i++ {
		args := make([]any, len(regs))
		for j, r := range regs {
			args[j] = r[i]
		}
		g.ins(format, args...)
	}
}

// xorRotate emits x = ror(x ^ y, n) for each of the four G functions.
func (g *gen) xorRotate(v variant, x, y [4]int, n int) {
	if v.sve2 {
		for i := range x {
			g.xar(x[i], y[i], n)
		}
		return
	}
	tmp := [4]int{scratch, scratch + 1, scratch + 2, scratch + 3}
	switch n {
	case 16:
		g.each("VEOR V%d.B16, V%d.B16, V%d.B16", y, x, x)
		g.each("VREV32 V%d.H8, V%d.H8", x, x)
	case 8:
		g.each("VEOR V%d.B16, V%d.B16, V%d.B16", y, x, x)
		g.each(fmt.Sprintf("VTBL V%d.B16, [V%%d.B16], V%%d.B16", rot8Table), x, x)
	default:
		g.each("VEOR V%d.B16, V%d.B16, V%d.B16", y, x, tmp)
		g.each(fmt.Sprintf("VSHL $%d, V%%d.S4, V%%d.S4", 32-n), tmp, x)
		g.each(fmt.Sprintf("VSRI $%d, V%%d.S4, V%%d.S4", n), tmp, x)
	}
}

// halfRound emits four interleaved G functions. The message add comes before
// the add of b so that it stays off the dependency chain through b.
func (g *gen) halfRound(v variant, a, b, c, d [4]int, m []int) {
	mx := [4]int{m[0], m[2], m[4], m[6]}
	my := [4]int{m[1], m[3], m[5], m[7]}
	g.addMsg(v, a, mx)
	g.each("VADD V%d.S4, V%d.S4, V%d.S4", b, a, a)
	g.xorRotate(v, d, a, 16)
	g.each("VADD V%d.S4, V%d.S4, V%d.S4", d, c, c)
	g.xorRotate(v, b, c, 12)
	g.addMsg(v, a, my)
	g.each("VADD V%d.S4, V%d.S4, V%d.S4", b, a, a)
	g.xorRotate(v, d, a, 8)
	g.each("VADD V%d.S4, V%d.S4, V%d.S4", d, c, c)
	g.xorRotate(v, b, c, 7)
}

// rounds emits the subroutine that runs the seven rounds on V0-V15.
func (g *gen) rounds(v variant) {
	g.section = "rounds" + v.name
	g.raw("")
	g.raw("// blake3rounds4%s runs the seven rounds of four compressions whose state", v.name)
	g.raw("// is in V0-V15. See gen_arm64.go for where it expects the message words.")
	g.raw("TEXT blake3rounds4%s<>(SB), NOSPLIT|NOFRAME, $0-0", v.name)
	var s [16]int
	for i := range s {
		s[i] = i
	}
	for r := 0; r < 7; r++ {
		g.comment("round %d, columns", r+1)
		g.halfRound(v,
			[4]int{0, 1, 2, 3}, [4]int{4, 5, 6, 7}, [4]int{8, 9, 10, 11}, [4]int{12, 13, 14, 15}, s[:8])
		g.comment("round %d, diagonals", r+1)
		g.halfRound(v,
			[4]int{0, 1, 2, 3}, [4]int{5, 6, 7, 4}, [4]int{10, 11, 8, 9}, [4]int{15, 12, 13, 14}, s[8:])
		var next [16]int
		for i := range next {
			next[i] = s[msgPermutation[i]]
		}
		s = next
	}
	g.ins("RET")
}

// transpose turns four registers holding four words of each lane into four
// registers holding one word of every lane, or the reverse. tmp names four
// scratch registers; dst may overlap src but not tmp.
func (g *gen) transpose(src, tmp, dst [4]int) {
	g.ins("VZIP1 V%d.S4, V%d.S4, V%d.S4", src[1], src[0], tmp[0])
	g.ins("VZIP2 V%d.S4, V%d.S4, V%d.S4", src[1], src[0], tmp[1])
	g.ins("VZIP1 V%d.S4, V%d.S4, V%d.S4", src[3], src[2], tmp[2])
	g.ins("VZIP2 V%d.S4, V%d.S4, V%d.S4", src[3], src[2], tmp[3])
	g.ins("VZIP1 V%d.D2, V%d.D2, V%d.D2", tmp[2], tmp[0], dst[0])
	g.ins("VZIP2 V%d.D2, V%d.D2, V%d.D2", tmp[2], tmp[0], dst[1])
	g.ins("VZIP1 V%d.D2, V%d.D2, V%d.D2", tmp[3], tmp[1], dst[2])
	g.ins("VZIP2 V%d.D2, V%d.D2, V%d.D2", tmp[3], tmp[1], dst[3])
}

// hash4 emits the kernel behind compressChunks and compressParents: it hashes
// the same number of blocks from each of four inputs.
func (g *gen) hash4(v variant) {
	g.section = "setup"
	g.raw("")
	g.raw("// func hash4%s(out *[4][8]uint32, p0, p1, p2, p3 *byte, blocks int, key *[8]uint32, counter, inc uint64, flags, flagsStart, flagsEnd uint32)", v.name)
	g.raw("TEXT ·hash4%s(SB), NOSPLIT, $%d-84", v.name, frameSize)
	g.ins("MOVD out+0(FP), R0")
	g.ins("MOVD p0+8(FP), R1")
	g.ins("MOVD p1+16(FP), R2")
	g.ins("MOVD p2+24(FP), R3")
	g.ins("MOVD p3+32(FP), R4")
	g.ins("MOVD blocks+40(FP), R5")
	g.ins("MOVD key+48(FP), R6")
	g.ins("MOVD counter+56(FP), R7")
	g.ins("MOVD inc+64(FP), R8")
	g.ins("MOVWU flags+72(FP), R9")
	g.ins("MOVWU flagsStart+76(FP), R10")
	g.ins("MOVWU flagsEnd+80(FP), R11")
	g.ins("MOVD $blake3consts<>(SB), R12")

	g.comment("V0-V7 carry the chaining values from block to block")
	for i := 0; i < 8; i++ {
		g.ins("VLD1R.P 4(R6), [V%d.S4]", i)
	}

	g.comment("lane i hashes with counter + i*inc")
	g.ins("VMOV R7, V8.D[0]")
	g.ins("ADD R8, R7, R7")
	g.ins("VMOV R7, V8.D[1]")
	g.ins("ADD R8, R7, R7")
	g.ins("VMOV R7, V9.D[0]")
	g.ins("ADD R8, R7, R7")
	g.ins("VMOV R7, V9.D[1]")
	g.ins("VUZP1 V9.S4, V8.S4, V10.S4")
	g.ins("VUZP2 V9.S4, V8.S4, V11.S4")
	g.ins("FMOVQ F10, %d(RSP)", frameCounterLo)
	g.ins("FMOVQ F11, %d(RSP)", frameCounterHi)
	if !v.sve2 {
		g.ins("FMOVQ %d(R12), F%d", constRot8, rot8Table)
	}

	g.comment("R13 holds the flags of the next block")
	g.ins("ORR R10, R9, R13")

	g.section = "block" + v.name
	g.label("loop")
	g.ins("CMP $1, R5")
	g.ins("BNE load")
	g.ins("ORR R11, R13, R13")
	g.label("load")
	g.comment("load a block from each input and transpose it into message words;")
	g.comment("V8-V15 are free until the state is set up below")
	for group := 0; group < 4; group++ {
		for lane := 0; lane < 4; lane++ {
			g.ins("FMOVQ %d(R%d), F%d", 16*group, 1+lane, 8+lane)
		}
		var dst [4]int
		var spill []int
		for i := range dst {
			w := 4*group + i
			if r, ok := v.msgReg(w); ok {
				dst[i] = r
			} else {
				dst[i] = scratch + i
				spill = append(spill, i)
			}
		}
		g.transpose([4]int{8, 9, 10, 11}, [4]int{12, 13, 14, 15}, dst)
		for _, i := range spill {
			g.ins("FMOVQ F%d, %d(RSP)", dst[i], v.msgSlot(4*group+i))
		}
	}
	for lane := 0; lane < 4; lane++ {
		g.ins("ADD $64, R%d, R%d", 1+lane, 1+lane)
	}

	g.comment("rows 2 and 3 of the state: IV, counter, block length, flags")
	g.ins("VLD1 (R12), [V8.B16, V9.B16, V10.B16, V11.B16]")
	g.ins("FMOVQ %d(RSP), F12", frameCounterLo)
	g.ins("FMOVQ %d(RSP), F13", frameCounterHi)
	g.ins("FMOVQ %d(R12), F14", constBlockLen)
	g.ins("VDUP R13, V15.S4")
	g.ins("BL blake3rounds4%s<>(SB)", v.name)
	g.comment("the next chaining value is the XOR of the two halves of the state")
	for i := 0; i < 8; i++ {
		g.ins("VEOR V%d.B16, V%d.B16, V%d.B16", 8+i, i, i)
	}
	g.ins("MOVD R9, R13")
	g.ins("SUBS $1, R5, R5")
	g.ins("BNE loop")

	g.section = "setup"
	g.comment("transpose the chaining values back and store one per lane")
	for half := 0; half < 2; half++ {
		src := [4]int{4 * half, 4*half + 1, 4*half + 2, 4*half + 3}
		g.transpose(src, [4]int{8, 9, 10, 11}, [4]int{12, 13, 14, 15})
		for lane := 0; lane < 4; lane++ {
			g.ins("FMOVQ F%d, %d(R0)", 12+lane, 32*lane+16*half)
		}
	}
	g.ins("RET")
}

// xof4 emits the kernel behind CompressBlocks: it compresses one node with
// four successive counters and stores all sixteen output words of each.
func (g *gen) xof4(v variant) {
	g.section = "xof" + v.name
	g.raw("")
	g.raw("// func xof4%s(out *[256]byte, n *Node)", v.name)
	g.raw("TEXT ·xof4%s(SB), NOSPLIT, $%d-16", v.name, frameSize)
	g.ins("MOVD out+0(FP), R0")
	g.ins("MOVD n+8(FP), R1")
	g.ins("MOVD $blake3consts<>(SB), R12")

	g.comment("every lane compresses the same block")
	g.ins("ADD $32, R1, R2")
	for i := 0; i < 16; i++ {
		if r, ok := v.msgReg(i); ok {
			g.ins("VLD1R.P 4(R2), [V%d.S4]", r)
		} else {
			g.ins("VLD1R.P 4(R2), [V8.S4]")
			g.ins("FMOVQ F8, %d(RSP)", v.msgSlot(i))
		}
	}

	g.comment("lane i uses counter + i; R2 points at the counter")
	g.ins("MOVD (R2), R3")
	g.ins("VDUP R3, V8.D2")
	g.ins("FMOVQ %d(R12), F9", constLanes01)
	g.ins("FMOVQ %d(R12), F10", constLanes23)
	g.ins("VADD V9.D2, V8.D2, V9.D2")
	g.ins("VADD V10.D2, V8.D2, V10.D2")
	g.ins("VUZP1 V10.S4, V9.S4, V12.S4")
	g.ins("VUZP2 V10.S4, V9.S4, V13.S4")
	g.comment("block length and flags follow the counter")
	g.ins("ADD $8, R2, R2")
	g.ins("VLD1R.P 4(R2), [V14.S4]")
	g.ins("VLD1R (R2), [V15.S4]")
	g.ins("MOVD R1, R2")
	for i := 0; i < 8; i++ {
		g.ins("VLD1R.P 4(R2), [V%d.S4]", i)
	}
	g.ins("VLD1 (R12), [V8.B16, V9.B16, V10.B16, V11.B16]")
	if !v.sve2 {
		g.ins("FMOVQ %d(R12), F%d", constRot8, rot8Table)
	}
	g.ins("BL blake3rounds4%s<>(SB)", v.name)

	g.comment("output words 0-7 are the XOR of the two halves of the state;")
	g.comment("words 8-15 are the second half XORed with the input chaining value")
	for i := 0; i < 8; i++ {
		g.ins("VEOR V%d.B16, V%d.B16, V%d.B16", 8+i, i, i)
	}
	g.ins("MOVD R1, R2")
	for i := 0; i < 8; i++ {
		g.ins("VLD1R.P 4(R2), [V16.S4]")
		g.ins("VEOR V16.B16, V%d.B16, V%d.B16", 8+i, 8+i)
	}
	g.comment("transpose the outputs back and store 64 bytes per lane")
	for group := 0; group < 4; group++ {
		src := [4]int{4 * group, 4*group + 1, 4*group + 2, 4*group + 3}
		g.transpose(src, [4]int{16, 17, 18, 19}, [4]int{20, 21, 22, 23})
		for lane := 0; lane < 4; lane++ {
			g.ins("FMOVQ F%d, %d(R0)", 20+lane, 64*lane+16*group)
		}
	}
	g.ins("RET")
}

func main() {
	out := flag.String("out", "compress_arm64.s", "file to write")
	stats := flag.Bool("stats", false, "print per-block instruction counts instead of writing the file")
	flag.Parse()

	g := &gen{counts: map[string]map[string]int{}}
	g.raw("// Code generated by gen_arm64.go. DO NOT EDIT.")
	g.raw("")
	g.raw("#include \"textflag.h\"")
	g.raw("")
	g.consts()
	for _, v := range []variant{neon, sve2} {
		g.rounds(v)
		g.hash4(v)
		g.xof4(v)
	}

	if *stats {
		fmt.Println("instructions per 64-byte block of four lanes (rounds + loop body):")
		for _, v := range []variant{neon, sve2} {
			total := map[string]int{}
			for _, section := range []string{"rounds" + v.name, "block" + v.name} {
				for kind, n := range g.counts[section] {
					total[kind] += n
				}
			}
			kinds := make([]string, 0, len(total))
			for kind := range total {
				kinds = append(kinds, kind)
			}
			sort.Strings(kinds)
			fmt.Printf("  %s:", v.name)
			for _, kind := range kinds {
				fmt.Printf(" %s=%d", kind, total[kind])
			}
			fmt.Println()
		}
		return
	}
	if err := os.WriteFile(*out, g.buf.Bytes(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
