package text_imgapproximator

//implementation of computationally-stable Vose's alias method
//for sampling from an array of weights.

//The weights are dimensionless; they are normalized by their sum.

import "core:math/rand"
import "core:math"
import "core:fmt"
import "core:testing"


Alias_Table :: struct {
	prob: []f32,
	alias: []int
}

Stack :: struct {
	//a stack of Atoms.
	top: ^Atom_Linked, //will be nil if empty
	len: int
}

Atom :: struct {
	p: f32,
	ind: int,
}
Atom_Linked :: struct {
	a: ^Atom,
	next: ^Atom_Linked
}
init_atom :: proc(p: f32, ind: int) -> ^Atom {
	out := new(Atom)
	out.p = p
	out.ind = ind
	return out
}
deinit_atom :: proc(a: ^Atom) {
	free(a)
}

init_stack :: proc() -> ^Stack {
	return new(Stack)
}
deinit_stack :: proc(s: ^Stack) {
	free(s)
}
push :: proc(s: ^Stack, a: ^Atom) {
	linked := new(Atom_Linked)
	linked.a = a
	linked.next = s.top
	s.top = linked
	s.len += 1
}
pop :: proc(s: ^Stack) -> ^Atom {
	assert(!stack_empty(s), "Attempt to pop from an empty stack")
	out := s.top.a
	free(s.top)
	s.top = s.top.next
	s.len -= 1
	return out
}
stack_empty :: proc(s: ^Stack) -> bool {
	if s.len == 0 do return true
	return false
}

normalize_vals :: proc(vals: []f32) {
	//normalizes vals in place to
	//"scaled probabilities"
	sum: f64 = 0
	for val in vals do sum += f64(val)
	n := len(vals)
	for i in 0..<n {
		vals[i] = vals[i]/f32(sum)*f32(n)
	}
}

init_alias_table :: proc(scaled_probs: []f32) -> ^Alias_Table {
	//Assumes scaled_probs is normalized properly;
	//if this is not the case, call normalize_vals,
	//or if we want to keep the original values unmodified,
	//allocate a copy of the vals and call normalize_vals
	//on that before passing here.
	n := len(scaled_probs)
	small := init_stack()
	large := init_stack()
	defer deinit_stack(small)
	defer deinit_stack(large)

	probs := make([]f32, n)
	alias := make([]int, n)
	for i in 0..<n {
		//initial pass to add
		//stuff to worklists
		prob := scaled_probs[i]
		a := init_atom(p=prob, ind=i)
		if prob < 1 {
			//add to small
			push(small, a)
		} else {
			//>=1; add to large
			push(large, a)
		}
	}
	for !stack_empty(small) && !stack_empty(large) {
		s := pop(small)
		l := pop(large)
		probs[s.ind] = s.p
		alias[s.ind] = l.ind

		//update the large probability
		l.p = (l.p + s.p) - 1

		if l.p < 1 {
			push(small, l)
		} else {
			push(large, l)
		}
		//we finalized the small atom;
		//we can free safely
		deinit_atom(s)
	}
	for !stack_empty(large) {
		//if stuff remains stable we should end up in this branch;
		//otherwise we end up in the next one
		l := pop(large)
		probs[l.ind] = 1
		deinit_atom(l)
	}
	for !stack_empty(small) {
		//same
		s := pop(small)
		probs[s.ind] = 1
		deinit_atom(s)
	}
	out := new(Alias_Table)
	out.prob = probs
	out.alias = alias
	return out
}

sample_from_tbl :: proc(tbl: ^Alias_Table) -> int {
	//sample an index using the table

	//first, an index of the table
	//[0,n) uniform
	table_ind := rand.int_range(0, len(tbl.prob))
	
	//[0,1) f32
	biased_coin_prob := rand.float32_range(0, 1)
	prob := tbl.prob[table_ind]
	if biased_coin_prob < prob {
		return table_ind
	} else {
		return tbl.alias[table_ind]
	}
}
deinit_alias_table :: proc(tbl: ^Alias_Table) {
	delete(tbl.prob)
	delete(tbl.alias)
	free(tbl)
}


@(test)
test_stack :: proc(t: ^testing.T) {
	s := init_stack()
	defer deinit_stack(s)
	a := init_atom(0.01, 2)
	b := init_atom(0.2, 1)
	defer free(a)
	defer free(b)
	push(s, a)
	push(s, b)
	pop1 := pop(s)
	pop2 := pop(s)
	testing.expect_value(t, pop1.p, f32(0.2))
	testing.expect_value(t, pop1.ind, 1)
	testing.expect_value(t, pop2.p, f32(0.01))
	testing.expect_value(t, pop2.ind, 2)
}

/*
Seems to work!
main :: proc() {
	vals := make([]f32, 4)
	defer delete(vals)
	vals[0] = 2 //7.1%
	vals[1] = 4 //14.3%
	vals[2] = 8 // 28.5%
	vals[3] = 14 //50%
	//skewed distribution
	normalize_vals(vals)
	tbl := init_alias_table(vals)
	defer deinit_alias_table(tbl)
	zeros := 0
	ones := 0
	twos := 0
	threes := 0
	n := 1000000
	for i in 0..<n {
		val := sample_from_tbl(tbl)
		if val == 0 do zeros += 1
		if val == 1 do ones += 1
		if val == 2 do twos += 1
		if val == 3 do threes += 1
	}
	//print results
	fmt.println("Proportion of 0: ", f32(zeros)/f32(n))
	fmt.println("Proportion of 1: ", f32(ones)/f32(n))
	fmt.println("Proportion of 2: ", f32(twos)/f32(n))
	fmt.println("Proportion of 3: ", f32(threes)/f32(n))	
}
*/
