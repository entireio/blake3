// Command avo generates guts/compress_amd64.s.
//
// It emits two families of kernels:
//
//   - AVX-512 kernels that hash sixteen (ZMM) or eight (YMM) inputs at once,
//     keeping the whole state and message schedule in registers;
//   - AVX2 kernels that hash eight inputs at once, with the message schedule
//     on the stack.
//
// These kernels read their inputs through a table of per-lane offsets, so
// a caller with fewer inputs than lanes can still use them: the spare lanes
// are pointed at the last valid input and their results are discarded.
package main

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

func main() {
	genGlobals()

	for _, lanes := range []int{16, 8} {
		genCompressChunksAVX512(lanes)
		genCompressParentsAVX512(lanes)
	}
	genCompressBlocksAVX512()

	genCompressChunksAVX2()
	genCompressParentsAVX2()
	genCompressBlocksAVX2()

	Generate()
}

var globals struct {
	iv           Mem
	seq          Mem
	seq64        Mem // for loadCounter
	blockLen     Mem
	shuffleRot8  Mem
	shuffleRot16 Mem
}

func genGlobals() {
	globals.iv = GLOBL("iv", RODATA|NOPTR)
	DATA(0*4, U32(0x6A09E667))
	DATA(1*4, U32(0xBB67AE85))
	DATA(2*4, U32(0x3C6EF372))
	DATA(3*4, U32(0xA54FF53A))

	globals.seq = GLOBL("seq", RODATA|NOPTR)
	for i := 0; i < 16; i++ {
		DATA(i*4, U32(i))
	}
	globals.seq64 = GLOBL("seq64", RODATA|NOPTR)
	for i := 0; i < 8; i++ {
		DATA(i*8, U64(i))
	}
	globals.blockLen = GLOBL("block_len", RODATA|NOPTR)
	DATA(0, U32(64))
	globals.shuffleRot8 = GLOBL("shuffle_rot8", RODATA|NOPTR)
	for i := 0; i < 8; i++ {
		DATA(i*4, U32(0x00030201+0x04040404*i))
	}
	globals.shuffleRot16 = GLOBL("shuffle_rot16", RODATA|NOPTR)
	for i := 0; i < 8; i++ {
		DATA(i*4, U32(0x01000302+0x04040404*i))
	}
}

// The BLAKE3 message schedule: round r+1 reads word permutation[i] of round r
// where round r read word i.
var permutation = [16]int{2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8}

// interleave4 transposes a 4x4 matrix of words inside every 128-bit lane:
// word j of dst[i] is word i of src[j]. tmp must not overlap src; dst may
// overlap src.
func interleave4(src, tmp, dst [4]Op) {
	VPUNPCKLDQ(src[1], src[0], tmp[0])
	VPUNPCKHDQ(src[1], src[0], tmp[1])
	VPUNPCKLDQ(src[3], src[2], tmp[2])
	VPUNPCKHDQ(src[3], src[2], tmp[3])
	VPUNPCKLQDQ(tmp[2], tmp[0], dst[0])
	VPUNPCKHQDQ(tmp[2], tmp[0], dst[1])
	VPUNPCKLQDQ(tmp[3], tmp[1], dst[2])
	VPUNPCKHQDQ(tmp[3], tmp[1], dst[3])
}

// -----------------------------------------------------------------------------
// AVX-512 kernels
// -----------------------------------------------------------------------------

var (
	xmm = [32]VecPhysical{X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31}
	ymm = [32]VecPhysical{Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15, Y16, Y17, Y18, Y19, Y20, Y21, Y22, Y23, Y24, Y25, Y26, Y27, Y28, Y29, Y30, Y31}
	zmm = [32]VecPhysical{Z0, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17, Z18, Z19, Z20, Z21, Z22, Z23, Z24, Z25, Z26, Z27, Z28, Z29, Z30, Z31}
)

// wide512 describes an AVX-512 kernel with one input per 32-bit lane of a ZMM
// (16 lanes) or YMM (8 lanes) register. With 32 vector registers the sixteen
// state words and the sixteen message words all stay in registers, and
// rotations are a single instruction.
//
// Registers are assigned by hand. Whichever half of the state is dead at a
// given point doubles as scratch space.
type wide512 struct {
	lanes int
	state [16]Op
	msg   [16]Op
}

func newWide512(lanes int) *wide512 {
	w := &wide512{lanes: lanes}
	for i := 0; i < 16; i++ {
		if lanes == 16 {
			w.msg[i] = w.vec(i)
			w.state[i] = w.vec(16 + i)
		} else {
			// The YMM kernel rotates with VPSHUFB where it can, whose VEX
			// encoding only reaches the low sixteen registers.
			w.state[i] = w.vec(i)
			w.msg[i] = w.vec(16 + i)
		}
	}
	return w
}

// vec returns physical register i at the kernel's vector width.
func (w *wide512) vec(i int) Op {
	if w.lanes == 16 {
		return zmm[i]
	}
	return ymm[i]
}

// bytes returns the size of a vector in bytes.
func (w *wide512) bytes() int { return w.lanes * 4 }

// half narrows v to half of the kernel's vector width.
func (w *wide512) half(v Op) Op {
	if w.lanes == 16 {
		return v.(VecPhysical).AsY()
	}
	return v.(VecPhysical).AsX()
}

// insertHalf replaces the upper half of v with m.
func (w *wide512) insertHalf(m Mem, v Op) {
	if w.lanes == 16 {
		VINSERTI64X4(Imm(1), m, v, v)
	} else {
		VINSERTI32X4(Imm(1), m, v, v)
	}
}

// laneOffsets fills a table with the byte offset of each lane's input:
// lane i reads input min(i, n-1), and inputs are 1<<shift bytes apart.
func (w *wide512) laneOffsets(n Register, shift int, scratch Op) Mem {
	offsets := AllocLocal(w.bytes())
	DECQ(n)
	VPBROADCASTD(n.(GPVirtual).As32(), scratch)
	VPMINUD(globals.seq, scratch, scratch)
	VPSLLD(Imm(uint64(shift)), scratch, scratch)
	VMOVDQU32(scratch, offsets)
	return offsets
}

// loadTransposed loads one 64-byte block from every lane and transposes
// them, leaving word i of every block in msg[i]. It clobbers state[8:].
//
// Each block is fetched in pieces half as wide as a 128-bit lane group, and
// pieces from two different inputs are combined by the load itself. That
// saves one of the shuffle stages a transpose of full-width rows would need,
// and shuffles are what the rounds are short of.
func (w *wide512) loadTransposed(base Register, offsets Mem) {
	tmp := w.state[8:]
	ra, rb := GP64(), GP64()
	// pair loads the piece at disp from lanes a and b into dst.
	pair := func(a, b, disp int, dst Op) {
		MOVL(offsets.Offset(4*a), ra.As32())
		MOVL(offsets.Offset(4*b), rb.As32())
		VMOVDQU32(Mem{Base: base, Index: ra, Scale: 1, Disp: disp}, w.half(dst))
		w.insertHalf(Mem{Base: base, Index: rb, Scale: 1, Disp: disp}, dst)
	}

	if w.lanes == 16 {
		// Rows hold eight words from lane i and eight from lane i+4. Two
		// in-lane stages gather words from four rows, and one cross-lane
		// stage joins the two groups of eight lanes.
		for half := 0; half < 2; half++ {
			out := w.msg[8*half : 8*half+8]
			for g := 0; g < 2; g++ {
				for i := 0; i < 4; i++ {
					pair(8*g+i, 8*g+i+4, 32*half, tmp[4*g+i])
				}
			}
			for g := 0; g < 2; g++ {
				rows := [4]Op(tmp[4*g : 4*g+4])
				interleave4(rows, [4]Op(out[4*g:4*g+4]), rows)
			}
			for k := 0; k < 4; k++ {
				VSHUFI32X4(Imm(0x88), tmp[4+k], tmp[k], out[k])
				VSHUFI32X4(Imm(0xDD), tmp[4+k], tmp[k], out[4+k])
			}
		}
		return
	}

	// Rows hold four words from lane i and four from lane i+4, so two
	// in-lane stages finish the job.
	for quarter := 0; quarter < 4; quarter++ {
		out := w.msg[4*quarter : 4*quarter+4]
		for i := 0; i < 4; i++ {
			pair(i, i+4, 16*quarter, tmp[i])
		}
		VPUNPCKLQDQ(tmp[1], tmp[0], tmp[4])
		VPUNPCKHQDQ(tmp[1], tmp[0], tmp[5])
		VPUNPCKLQDQ(tmp[3], tmp[2], tmp[6])
		VPUNPCKHQDQ(tmp[3], tmp[2], tmp[7])
		VSHUFPS(Imm(0x88), tmp[6], tmp[4], out[0])
		VSHUFPS(Imm(0xDD), tmp[6], tmp[4], out[1])
		VSHUFPS(Imm(0x88), tmp[7], tmp[5], out[2])
		VSHUFPS(Imm(0xDD), tmp[7], tmp[5], out[3])
	}
}

// rounds performs the seven rounds of the compression function.
func (w *wide512) rounds() {
	// Intel executes 512-bit integer operations on two ports, and rotations
	// on only one of them, which the additions and XORs balance out. YMM code
	// has a third port, but rotations still have just the one; rotating by
	// whole bytes with a shuffle moves half of them off it.
	rotate := func(n uint64, v Op) {
		switch {
		case w.lanes == 8 && n == 16:
			VPSHUFB(globals.shuffleRot16, v, v)
		case w.lanes == 8 && n == 8:
			VPSHUFB(globals.shuffleRot8, v, v)
		default:
			VPRORD(Imm(n), v, v)
		}
	}
	g := func(a, b, c, d, mx, my Op) {
		// Adding the message word first keeps it off the critical path: b is
		// the last input of each G to become available.
		VPADDD(mx, a, a)
		VPADDD(b, a, a)
		VPXORD(a, d, d)
		rotate(16, d)
		VPADDD(d, c, c)
		VPXORD(c, b, b)
		rotate(12, b)
		VPADDD(my, a, a)
		VPADDD(b, a, a)
		VPXORD(a, d, d)
		rotate(8, d)
		VPADDD(d, c, c)
		VPXORD(c, b, b)
		rotate(7, b)
	}

	vs, mv := w.state, w.msg
	for i := 0; i < 7; i++ {
		Comment(fmt.Sprintf("Round %v", i+1))
		g(vs[0], vs[4], vs[8], vs[12], mv[0], mv[1])
		g(vs[1], vs[5], vs[9], vs[13], mv[2], mv[3])
		g(vs[2], vs[6], vs[10], vs[14], mv[4], mv[5])
		g(vs[3], vs[7], vs[11], vs[15], mv[6], mv[7])
		g(vs[0], vs[5], vs[10], vs[15], mv[8], mv[9])
		g(vs[1], vs[6], vs[11], vs[12], mv[10], mv[11])
		g(vs[2], vs[7], vs[8], vs[13], mv[12], mv[13])
		g(vs[3], vs[4], vs[9], vs[14], mv[14], mv[15])

		var next [16]Op
		for j := range next {
			next[j] = mv[permutation[j]]
		}
		mv = next
	}
}

// loadCounters sets lo and hi to the low and high words of counter+lane. It
// clobbers counter.
func (w *wide512) loadCounters(counter GPVirtual, lo, hi Op) {
	VPBROADCASTD(counter.As32(), lo)
	VPADDD(globals.seq, lo, lo)
	// set a 1 bit in K1 for each overflowed counter in lo
	VPCMPUD(Imm(1), globals.seq, lo, K1)
	// add 1 to each counter in hi for each 1 bit in K1
	SHRQ(Imm(32), counter)
	VPBROADCASTD(counter.As32(), hi)
	VPADDD_BCST(globals.seq.Offset(4), hi, K1, hi)
}

// storeCVs transposes state[:8], which must hold one chaining value per
// lane, and writes the lanes' chaining values to out in order. It clobbers
// msg.
func (w *wide512) storeCVs(out Register) {
	t, u := w.msg[:8], w.msg[8:]
	// Lane q of u[r] holds words 0-3 of input 4q+r, and lane q of u[4+r]
	// holds its words 4-7.
	interleave4([4]Op(w.state[0:4]), [4]Op(t[0:4]), [4]Op(u[0:4]))
	interleave4([4]Op(w.state[4:8]), [4]Op(t[4:8]), [4]Op(u[4:8]))

	if w.lanes == 8 {
		for r := 0; r < 4; r++ {
			VSHUFI32X4(Imm(0x0), u[4+r], u[r], t[r])
			VSHUFI32X4(Imm(0x3), u[4+r], u[r], t[4+r])
		}
		for i := 0; i < 8; i++ {
			VMOVDQU32(t[i], Mem{Base: out, Disp: 32 * i})
		}
		return
	}

	for r := 0; r < 4; r++ {
		// t[r]   = low and high halves of inputs r and 4+r
		// t[4+r] = low and high halves of inputs 8+r and 12+r
		VSHUFI32X4(Imm(0x44), u[4+r], u[r], t[r])
		VSHUFI32X4(Imm(0xEE), u[4+r], u[r], t[4+r])
	}
	// Each store covers two consecutive inputs.
	store := func(a, b Op, imm uint64, input int) {
		VSHUFI32X4(Imm(imm), b, a, u[0])
		VMOVDQU32(u[0], Mem{Base: out, Disp: 32 * input})
	}
	for g := 0; g < 2; g++ {
		for r := 0; r < 4; r += 2 {
			store(t[4*g+r], t[4*g+r+1], 0x88, 8*g+r)
			store(t[4*g+r], t[4*g+r+1], 0xDD, 8*g+r+4)
		}
	}
}

func genCompressChunksAVX512(lanes int) {
	TEXT(fmt.Sprintf("compressChunks%dAVX512", lanes), NOSPLIT,
		fmt.Sprintf("func(cvs *[%d][8]uint32, buf *byte, n uint64, key *[8]uint32, counter uint64, flags uint32)", lanes))
	w := newWide512(lanes)
	cvs := Load(Param("cvs"), GP64())
	buf := Load(Param("buf"), GP64())
	n := Load(Param("n"), GP64())
	key := Mem{Base: Load(Param("key"), GP64())}
	counter := Load(Param("counter"), GP64()).(GPVirtual)
	flags := Load(Param("flags"), GP32())

	Comment("Locate each lane's chunk")
	offsets := w.laneOffsets(n, 10, w.state[8])

	Comment("Initialize counter")
	counterLo := AllocLocal(w.bytes())
	counterHi := AllocLocal(w.bytes())
	w.loadCounters(counter, w.state[8], w.state[9])
	VMOVDQU32(w.state[8], counterLo)
	VMOVDQU32(w.state[9], counterHi)

	Comment("Initialize flags")
	chunkFlags := AllocLocal(16 * 4)
	for i := 0; i < 16; i++ {
		MOVL(flags, chunkFlags.Offset(4*i))
	}
	ORL(Imm(1), chunkFlags.Offset(0*4))
	ORL(Imm(2), chunkFlags.Offset(15*4))

	Comment("Load key")
	for i := 0; i < 8; i++ {
		VPBROADCASTD(key.Offset(i*4), w.state[i])
	}

	Comment("Loop index")
	loop := GP64()
	XORQ(loop, loop)
	Label("loop")

	Comment("Load transposed block")
	w.loadTransposed(buf, offsets)
	ADDQ(Imm(64), buf)

	Comment("Reload state vectors (other than CVs)")
	for i := 0; i < 4; i++ {
		VPBROADCASTD(globals.iv.Offset(i*4), w.state[8+i])
	}
	VMOVDQU32(counterLo, w.state[12])
	VMOVDQU32(counterHi, w.state[13])
	VPBROADCASTD(globals.blockLen, w.state[14])
	VPBROADCASTD(chunkFlags.Idx(loop, 4), w.state[15])

	w.rounds()

	Comment("Finalize CVs")
	for i := 0; i < 8; i++ {
		VPXORD(w.state[i+8], w.state[i], w.state[i])
	}

	Comment("Loop")
	INCQ(loop)
	CMPQ(loop, U32(16))
	JNE(LabelRef("loop"))

	Comment("Finished; transpose CVs")
	w.storeCVs(cvs)

	VZEROUPPER()
	RET()
}

func genCompressParentsAVX512(lanes int) {
	TEXT(fmt.Sprintf("compressParents%dAVX512", lanes), NOSPLIT,
		fmt.Sprintf("func(parents *[%d][8]uint32, cvs *byte, n uint64, key *[8]uint32, flags uint32)", lanes))
	w := newWide512(lanes)
	parents := Load(Param("parents"), GP64())
	cvs := Load(Param("cvs"), GP64())
	n := Load(Param("n"), GP64())
	key := Mem{Base: Load(Param("key"), GP64())}
	flags := Load(Param("flags"), GP32())

	Comment("Locate each lane's pair of chaining values")
	offsets := w.laneOffsets(n, 6, w.state[8])

	Comment("Load transposed block")
	w.loadTransposed(cvs, offsets)

	Comment("Initialize state vectors")
	for i := 0; i < 8; i++ {
		VPBROADCASTD(key.Offset(i*4), w.state[i])
	}
	for i := 0; i < 4; i++ {
		VPBROADCASTD(globals.iv.Offset(i*4), w.state[8+i])
	}
	VPXORD(w.state[12], w.state[12], w.state[12]) // counter is zero for parents
	VPXORD(w.state[13], w.state[13], w.state[13])
	VPBROADCASTD(globals.blockLen, w.state[14])
	VPBROADCASTD(flags, w.state[15])

	w.rounds()

	Comment("Finalize CVs")
	for i := 0; i < 8; i++ {
		VPXORD(w.state[i+8], w.state[i], w.state[i])
	}
	w.storeCVs(parents)

	VZEROUPPER()
	RET()
}

func genCompressBlocksAVX512() {
	TEXT("compressBlocksAVX512", NOSPLIT, "func(out *[1024]byte, block *[16]uint32, cv *[8]uint32, counter uint64, blockLen uint32, flags uint32)")
	w := newWide512(16)
	out := Load(Param("out"), GP64())
	block := Mem{Base: Load(Param("block"), GP64())}
	cv := Mem{Base: Load(Param("cv"), GP64())}
	counter := Load(Param("counter"), GP64()).(GPVirtual)
	blockLen := Load(Param("blockLen"), GP32())
	flags := Load(Param("flags"), GP32())

	Comment("Initialize block vectors")
	for i := range w.msg {
		VPBROADCASTD(block.Offset(i*4), w.msg[i])
	}

	Comment("Initialize state vectors")
	for i := 0; i < 8; i++ {
		VPBROADCASTD(cv.Offset(i*4), w.state[i])
	}
	for i := 0; i < 4; i++ {
		VPBROADCASTD(globals.iv.Offset(i*4), w.state[8+i])
	}
	w.loadCounters(counter, w.state[12], w.state[13])
	VPBROADCASTD(blockLen, w.state[14])
	VPBROADCASTD(flags, w.state[15])

	w.rounds()

	Comment("Finalize CVs")
	for i := 0; i < 8; i++ {
		VPXORD(w.state[i+8], w.state[i], w.state[i])
	}
	for i := 0; i < 8; i++ {
		VPXORD_BCST(cv.Offset(i*4), w.state[8+i], w.state[8+i])
	}

	Comment("Transpose and store the sixteen output blocks")
	// Lane q of u[4g+r] holds words 4g..4g+3 of output block 4q+r.
	var u [16]Op
	for g := 0; g < 4; g++ {
		words := [4]Op(w.state[4*g : 4*g+4])
		interleave4(words, [4]Op(w.msg[4*g:4*g+4]), words)
		copy(u[4*g:], words[:])
	}
	for r := 0; r < 4; r++ {
		p := w.msg[4*r : 4*r+4]
		// Transposing the 128-bit lanes of u[r], u[4+r], u[8+r], u[12+r]
		// yields output blocks r, 4+r, 8+r and 12+r.
		VSHUFI32X4(Imm(0x88), u[4+r], u[r], p[0])
		VSHUFI32X4(Imm(0xDD), u[4+r], u[r], p[1])
		VSHUFI32X4(Imm(0x88), u[12+r], u[8+r], p[2])
		VSHUFI32X4(Imm(0xDD), u[12+r], u[8+r], p[3])
		row := u[r]
		for q, sel := range []struct {
			a, b Op
			imm  uint64
		}{
			{p[0], p[2], 0x88},
			{p[1], p[3], 0x88},
			{p[0], p[2], 0xDD},
			{p[1], p[3], 0xDD},
		} {
			VSHUFI32X4(Imm(sel.imm), sel.b, sel.a, row)
			VMOVDQU32(row, Mem{Base: out, Disp: 64 * (4*q + r)})
		}
	}

	VZEROUPPER()
	RET()
}

// -----------------------------------------------------------------------------
// AVX2 kernels
// -----------------------------------------------------------------------------

// laneOffsetsAVX2 fills a table with the byte offset of each lane's input:
// lane i reads input min(i, n-1), and inputs are 1<<shift bytes apart.
func laneOffsetsAVX2(n Register, shift int, scratch VecVirtual) Mem {
	offsets := AllocLocal(32)
	DECQ(n)
	MOVL(n.(GPVirtual).As32(), offsets)
	VPBROADCASTD(offsets, scratch)
	VPMINUD(globals.seq, scratch, scratch)
	VPSLLD(Imm(uint64(shift)), scratch, scratch)
	VMOVDQU(scratch, offsets)
	return offsets
}

// broadcastAVX2 fills dst with copies of the 32-bit argument name.
func broadcastAVX2(name string, dst VecVirtual) {
	r := Load(Param(name), GP32())
	VMOVD(r, dst.AsX())
	VPBROADCASTD(dst.AsX(), dst)
}

// loadTransposedAVX2 loads one 64-byte block from each of eight lanes and
// stores the transposed words in mv, using tmp as scratch space.
//
// Rows hold four words from lane i and four from lane i+4, combined by the
// load itself, so two in-lane shuffle stages complete the transpose.
func loadTransposedAVX2(base Register, offsets Mem, mv [16]Mem, tmp []VecVirtual) {
	ra, rb := GP64(), GP64()
	for quarter := 0; quarter < 4; quarter++ {
		for i := 0; i < 4; i++ {
			MOVL(offsets.Offset(4*i), ra.As32())
			MOVL(offsets.Offset(4*(i+4)), rb.As32())
			VMOVDQU(Mem{Base: base, Index: ra, Scale: 1, Disp: 16 * quarter}, tmp[i].AsX())
			VINSERTI128(Imm(1), Mem{Base: base, Index: rb, Scale: 1, Disp: 16 * quarter}, tmp[i], tmp[i])
		}
		VPUNPCKLQDQ(tmp[1], tmp[0], tmp[4])
		VPUNPCKHQDQ(tmp[1], tmp[0], tmp[5])
		VPUNPCKLQDQ(tmp[3], tmp[2], tmp[6])
		VPUNPCKHQDQ(tmp[3], tmp[2], tmp[7])
		VSHUFPS(Imm(0x88), tmp[6], tmp[4], tmp[0])
		VSHUFPS(Imm(0xDD), tmp[6], tmp[4], tmp[1])
		VSHUFPS(Imm(0x88), tmp[7], tmp[5], tmp[2])
		VSHUFPS(Imm(0xDD), tmp[7], tmp[5], tmp[3])
		for i := 0; i < 4; i++ {
			VMOVDQU(tmp[i], mv[4*quarter+i])
		}
	}
}

func genCompressBlocksAVX2() {
	TEXT("compressBlocksAVX2", NOSPLIT, "func(out *[512]byte, block *[16]uint32, cv *[8]uint32, counter uint64, blockLen uint32, flags uint32)")
	out := Mem{Base: Load(Param("out"), GP64())}
	block := Mem{Base: Load(Param("block"), GP64())}
	cv := Mem{Base: Load(Param("cv"), GP64())}
	counter, _ := Param("counter").Resolve()

	var vs [16]VecVirtual
	var mv [16]Mem
	for i := range vs {
		vs[i] = YMM()
		mv[i] = AllocLocal(32)
	}

	Comment("Load block")
	for i := 0; i < 16; i++ {
		VPBROADCASTD(block.Offset(i*4), vs[0])
		VMOVDQU(vs[0], mv[i])
	}

	Comment("Initialize state vectors")
	for i, v := range vs {
		switch i {
		case 0, 1, 2, 3, 4, 5, 6, 7: // cv
			VPBROADCASTD(cv.Offset(i*4), v)
		case 8, 9, 10, 11: // iv
			VPBROADCASTD(globals.iv.Offset((i-8)*4), v)
		case 12: // counter
			loadCounter(counter.Addr, vs[12:14], vs[14:16])
		case 14: // blockLen
			broadcastAVX2("blockLen", v)
		case 15: // flags
			broadcastAVX2("flags", v)
		}
	}

	performRoundsAVX2(vs, mv)

	Comment("Finalize CVs")
	for i := 8; i < 16; i++ {
		VMOVDQU(vs[i], mv[i])
	}
	for i := range vs[:8] {
		VPXOR(vs[i], vs[i+8], vs[i])
	}
	transpose(vs[:8], vs[8:])
	for i, v := range vs[8:] {
		VMOVDQU(v, out.Offset(i*64))
	}
	for i := 8; i < 16; i++ {
		VMOVDQU(mv[i], vs[i])
	}
	for i, v := range vs[8:] {
		VPBROADCASTD(cv.Offset(i*4), vs[0])
		VPXOR(vs[0], v, v)
	}
	transpose(vs[8:], vs[:8])
	for i, v := range vs[:8] {
		VMOVDQU(v, out.Offset(i*64+32))
	}

	// See https://community.intel.com/t5/Intel-ISA-Extensions/What-is-the-status-of-VZEROUPPER-use/m-p/1098375
	VZEROUPPER()
	RET()
}

func genCompressChunksAVX2() {
	TEXT("compressChunks8AVX2", NOSPLIT, "func(cvs *[8][8]uint32, buf *byte, n uint64, key *[8]uint32, counter uint64, flags uint32)")
	cvs := Mem{Base: Load(Param("cvs"), GP64())}
	buf := Load(Param("buf"), GP64())
	n := Load(Param("n"), GP64())
	key := Mem{Base: Load(Param("key"), GP64())}
	counter, _ := Param("counter").Resolve()

	var vs [16]VecVirtual
	var mv [16]Mem
	for i := range vs {
		vs[i] = YMM()
		mv[i] = AllocLocal(32)
	}

	Comment("Locate each lane's chunk")
	offsets := laneOffsetsAVX2(n, 10, vs[8])

	Comment("Load key")
	for i := 0; i < 8; i++ {
		VPBROADCASTD(key.Offset(i*4), vs[i])
	}

	Comment("Initialize counter")
	counterLo := AllocLocal(32)
	counterHi := AllocLocal(32)
	loadCounter(counter.Addr, vs[12:14], vs[14:16])
	VMOVDQU(vs[12], counterLo)
	VMOVDQU(vs[13], counterHi)

	Comment("Initialize flags")
	chunkFlags := AllocLocal(16 * 4)
	broadcastAVX2("flags", vs[14])
	VMOVDQU(vs[14], chunkFlags.Offset(0*32))
	VMOVDQU(vs[14], chunkFlags.Offset(1*32))
	ORL(Imm(1), chunkFlags.Offset(0*4))
	ORL(Imm(2), chunkFlags.Offset(15*4))

	Comment("Loop index")
	loop := GP64()
	XORQ(loop, loop)
	Label("loop")

	Comment("Load transposed block")
	loadTransposedAVX2(buf, offsets, mv, vs[8:])
	ADDQ(Imm(64), buf)

	Comment("Reload state vectors (other than CVs)")
	for i := 0; i < 4; i++ {
		VPBROADCASTD(globals.iv.Offset(i*4), vs[8+i])
	}
	VMOVDQU(counterLo, vs[12])
	VMOVDQU(counterHi, vs[13])
	VPBROADCASTD(globals.blockLen, vs[14])
	VPBROADCASTD(chunkFlags.Idx(loop, 4), vs[15])

	performRoundsAVX2(vs, mv)

	Comment("Finalize CVs")
	for i := range vs[:8] {
		VPXOR(vs[i], vs[i+8], vs[i])
	}

	Comment("Loop")
	INCQ(loop)
	CMPQ(loop, U32(16))
	JNE(LabelRef("loop"))

	Comment("Finished; transpose CVs")
	transpose(vs[:8], vs[8:])
	for i, v := range vs[8:] {
		VMOVDQU(v, cvs.Offset(i*32))
	}

	VZEROUPPER()
	RET()
}

func genCompressParentsAVX2() {
	TEXT("compressParents8AVX2", NOSPLIT, "func(parents *[8][8]uint32, cvs *byte, n uint64, key *[8]uint32, flags uint32)")
	parents := Mem{Base: Load(Param("parents"), GP64())}
	cvs := Load(Param("cvs"), GP64())
	n := Load(Param("n"), GP64())
	key := Mem{Base: Load(Param("key"), GP64())}

	var vs [16]VecVirtual
	var mv [16]Mem
	for i := range vs {
		vs[i] = YMM()
		mv[i] = AllocLocal(32)
	}

	Comment("Locate each lane's pair of chaining values")
	offsets := laneOffsetsAVX2(n, 6, vs[8])

	Comment("Load transposed block")
	loadTransposedAVX2(cvs, offsets, mv, vs[8:])

	Comment("Initialize state vectors")
	for i, v := range vs {
		switch i {
		case 0, 1, 2, 3, 4, 5, 6, 7: // cv
			VPBROADCASTD(key.Offset(i*4), v)
		case 8, 9, 10, 11: // iv
			VPBROADCASTD(globals.iv.Offset((i-8)*4), v)
		case 12, 13: // counter
			VPXOR(v, v, v)
		case 14: // blockLen
			VPBROADCASTD(globals.blockLen, v)
		case 15: // flags
			broadcastAVX2("flags", v)
		}
	}

	performRoundsAVX2(vs, mv)

	Comment("Finalize CVs")
	for i := range vs[:8] {
		VPXOR(vs[i], vs[i+8], vs[i])
	}
	transpose(vs[:8], vs[8:])
	for i, v := range vs[8:] {
		VMOVDQU(v, parents.Offset(i*32))
	}

	VZEROUPPER()
	RET()
}

func performRoundsAVX2(sv [16]VecVirtual, mv [16]Mem) {
	spillMem := AllocLocal(32)
	tmp := sv[8]
	g := func(a, b, c, d VecVirtual, mx, my Mem) {
		// Helper function for performing rotations. Also manages c, tmp and
		// spillMem: if c == tmp, we need to spill and reload c using spillMem.
		rotr := func(v VecVirtual, n uint64, dst VecVirtual) {
			switch n {
			case 8, 16:
				shuf := [...]Mem{8: globals.shuffleRot8, 16: globals.shuffleRot16}[n]
				VPSHUFB(shuf, v, dst)
				if c == tmp {
					VMOVDQU(spillMem, c)
				}
			case 7, 12:
				if c == tmp {
					VMOVDQU(c, spillMem)
				}
				VPSRLD(Imm(n), v, tmp)
				VPSLLD(Imm(32-n), v, dst)
				VPOR(dst, tmp, dst)
			}
		}

		VPADDD(mx, a, a)
		VPADDD(a, b, a)
		VPXOR(d, a, d)
		rotr(d, 16, d)
		VPADDD(c, d, c)
		VPXOR(b, c, b)
		rotr(b, 12, b)
		VPADDD(my, a, a)
		VPADDD(a, b, a)
		VPXOR(d, a, d)
		rotr(d, 8, d)
		VPADDD(c, d, c)
		VPXOR(b, c, b)
		rotr(b, 7, b)
	}

	VMOVDQU(sv[8], spillMem) // spill
	for i := 0; i < 7; i++ {
		Comment(fmt.Sprintf("Round %v", i+1))
		g(sv[0], sv[4], sv[8], sv[12], mv[0], mv[1])
		g(sv[1], sv[5], sv[9], sv[13], mv[2], mv[3])
		g(sv[2], sv[6], sv[10], sv[14], mv[4], mv[5])
		g(sv[3], sv[7], sv[11], sv[15], mv[6], mv[7])
		g(sv[0], sv[5], sv[10], sv[15], mv[8], mv[9])
		g(sv[1], sv[6], sv[11], sv[12], mv[10], mv[11])
		g(sv[2], sv[7], sv[8], sv[13], mv[12], mv[13])
		g(sv[3], sv[4], sv[9], sv[14], mv[14], mv[15])

		var next [16]Mem
		for j := range next {
			next[j] = mv[permutation[j]]
		}
		mv = next
	}
	VMOVDQU(spillMem, sv[8]) // reload
}

func loadCounter(counter Mem, dst, scratch []VecVirtual) {
	// fill dst[0] and dst[1] with counter + 0,1,2,3,4,5,6,7, then transpose so
	// that dst[0] contains low 32 bits and dst[1] contains high 32 bits.
	VPBROADCASTQ(counter, dst[0])
	VPBROADCASTQ(counter, dst[1])
	VPADDQ(globals.seq64.Offset(0*4), dst[0], dst[0])
	VPADDQ(globals.seq64.Offset(8*4), dst[1], dst[1])
	VPUNPCKLDQ(dst[1], dst[0], scratch[0])
	VPUNPCKHDQ(dst[1], dst[0], scratch[1])
	VPUNPCKLDQ(scratch[1], scratch[0], dst[0])
	VPUNPCKHDQ(scratch[1], scratch[0], dst[1])
	const perm = 0<<0 | 2<<2 | 1<<4 | 3<<6
	VPERMQ(Imm(perm), dst[0], dst[0])
	VPERMQ(Imm(perm), dst[1], dst[1])
}

func transpose(src, dst []VecVirtual) {
	// interleave uint32s
	for i := 0; i < 8; i += 2 {
		VPUNPCKLDQ(src[i+1], src[i], dst[i+0])
		VPUNPCKHDQ(src[i+1], src[i], dst[i+1])
	}
	// interleave groups of two uint32s
	for i := 0; i < 4; i++ {
		j := i*2 - i%2 // j := 0,1,4,5
		VPUNPCKLQDQ(dst[j+2], dst[j], src[i*2+0])
		VPUNPCKHQDQ(dst[j+2], dst[j], src[i*2+1])
	}
	// interleave groups of four uint32s
	for i := 0; i < 4; i++ {
		VPERM2I128(Imm(0x20), src[i+4], src[i], dst[i+0])
		VPERM2I128(Imm(0x31), src[i+4], src[i], dst[i+4])
	}
}
