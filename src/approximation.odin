package text_imgapproximator

//implementation of the stochastic algorithm
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:fmt"
import "core:slice"
import "core:strings"
import "core:strconv"

import "core:thread"
import "core:sync"
import "base:runtime"

//structs
Frame :: struct {
	x0, y0, w, h: int
}

One_Channel_Img :: struct {
	w,h: int,
	data: []int
}
One_Channel_Img_f32 :: struct { //definitely unidiomatic but im lazy
	w,h: int,
	data: []f32 
}

Iter_Ctx :: struct {
    curr: ^Image_Data,
    target: ^Image_Data,
    resid: One_Channel_Img,
    sq_resid: One_Channel_Img,

    saliency_tbl: ^Alias_Table,
    cache: Multiscale_SDF_Cache,

    w, h: int,
    R: f32,

    //outputs written per proposal index i
    props: []Glyph,
    improvements: []f32,
	allocator: runtime.Allocator,

    proposal_count: int,
}

Thread_Pool :: struct {
    threads: []^thread.Thread,
    locals: []Worker_Local,

    mu: sync.Mutex,
    cv: sync.Cond,
    generation: int,
    stop: bool,

    next_job: int,
    done: int,

    //pointer to current iteration context (valid while generation active)
    ctx: ^Iter_Ctx,
}

Worker_Local :: struct {
    id: int,
    scale_vals: []f32,
    scores: []f32,
    bitmaps: []Glyph_Bitmap,
}

pool_init :: proc(num_threads: int, cache: Multiscale_SDF_Cache) -> ^Thread_Pool {
	pool := new(Thread_Pool)
    pool.threads = make([]^thread.Thread, num_threads)
    pool.locals = make([]Worker_Local, num_threads)

    //init worker locals once
    for i in 0..<num_threads {
        pool.locals[i].id = i
        pool.locals[i].scale_vals = make([]f32, len(cache.px_heights))
        pool.locals[i].scores = make([]f32, len(cache.valid_codepoints))
		pool.locals[i].bitmaps = make([]Glyph_Bitmap, len(cache.valid_codepoints))
    }

    //spawn threads once
    for i in 0..<num_threads {
        pool.threads[i] = thread.create(worker_main)
		pool.threads[i].user_index=i
		pool.threads[i].data = pool //so that they can access their data
		thread.start(pool.threads[i])
    }
    return pool
}

pool_deinit :: proc(pool: ^Thread_Pool) {
    sync.lock(&pool.mu)
    pool.stop = true
    pool.generation += 1
    sync.cond_broadcast(&pool.cv)
    sync.unlock(&pool.mu)

    for t in pool.threads {
        thread.join(t)
		thread.destroy(t)
    }

    //free locals scratch
    for i in 0..<len(pool.locals) {
        delete(pool.locals[i].scale_vals)
        delete(pool.locals[i].scores)
    }
    delete(pool.locals)
    delete(pool.threads)
	free(pool)
}

worker_main :: proc(t: ^thread.Thread) {
	local_gen := 0
	pool := (cast(^Thread_Pool)t.data)
    wl := &pool.locals[t.user_index]

    for {
        //wait for work
        sync.lock(&pool.mu)
        for !pool.stop && pool.generation == local_gen {
            sync.cond_wait(&pool.cv, &pool.mu)
        }
        if pool.stop {
            sync.unlock(&pool.mu)
            break
        }
        local_gen = pool.generation
        ctx := pool.ctx
        sync.unlock(&pool.mu)

        //process jobs
        for {
            i := int(sync.atomic_add(&pool.next_job, 1))
            if i >= ctx.proposal_count do break
            compute_proposal(ctx, wl, i)
        }

        // signal completion
        if sync.atomic_add(&pool.done, 1) + 1 == len(pool.threads) {
            sync.lock(&pool.mu)
            sync.cond_signal(&pool.cv)
            sync.unlock(&pool.mu)
        }
    }
}

compute_proposal :: proc(ctx: ^Iter_Ctx, wl: ^Worker_Local, i: int) {
	//computes the proposal from context ctx and inserts it at index i
	//in the context
	curr := ctx.curr
    target := ctx.target
    resid := ctx.resid
    sq_resid := ctx.sq_resid
    saliency_tbl := ctx.saliency_tbl
    cache := ctx.cache
    w := ctx.w
	h := ctx.h
	R := ctx.R

	//sample position
	id_pos := sample_from_tbl(saliency_tbl)
	pos_x := id_pos % w 
	pos_y := id_pos / w 

	//sample angle
	theta := sample_rotation()

	for j in 0..<len(cache.px_heights) {
		dim := get_dimensions_from_cache(cache, cache.px_heights[j])
		frame := frame_from_dimensions(pos_x,pos_y,dim)
		wl.scale_vals[j] = energy_density_frame(sq_resid, frame)
	}
	numworst_ph := len(cache.px_heights)/3
	if numworst_ph == 0 do numworst_ph += 1
	softmax(wl.scale_vals, R, numworst_ph)
	normalize_vals(wl.scale_vals)
	tbl_scale := init_alias_table(wl.scale_vals)
			
	//sample and put the height into heights[i]
	height := cache.px_heights[sample_from_tbl(tbl_scale)]
	deinit_alias_table(tbl_scale)

	auxiliary_bitmaps := make([]Glyph_Bitmap, len(cache.valid_codepoints), context.temp_allocator)
	for j in 0..<len(auxiliary_bitmaps) {
		auxiliary_bitmaps[j] = get_bitmap(cache, height, 
										cache.valid_codepoints[j],
										theta, 1.0, context.temp_allocator)
		//compute score
		wl.scores[j] = glyph_score(resid, auxiliary_bitmaps[j], pos_x, pos_y)
	}
	numworst := len(wl.scores)/3 //might be 0 if we have small dictionary; add 1
	if numworst == 0 do numworst += 1
	softmax(wl.scores, R, numworst)
	normalize_vals(wl.scores)
	tbl_scores := init_alias_table(wl.scores)
	bitmap_index := sample_from_tbl(tbl_scores)
	deinit_alias_table(tbl_scores)
	//copy bmap into props
	if ctx.props[i].bmap != nil do delete(ctx.props[i].bmap, ctx.allocator)
	ctx.props[i].bmap = make_slice([]u8, len(auxiliary_bitmaps[bitmap_index].mask), allocator=ctx.allocator) //make with the parent allocator
	mem.copy(&ctx.props[i].bmap[0], &auxiliary_bitmaps[bitmap_index].mask[0], len(auxiliary_bitmaps[bitmap_index].mask))
	ctx.props[i].w = auxiliary_bitmaps[bitmap_index].w
	ctx.props[i].h = auxiliary_bitmaps[bitmap_index].h
	ctx.props[i].x = pos_x
	ctx.props[i].y = pos_y
	optim_color := optimal_color(curr, target, auxiliary_bitmaps[bitmap_index], pos_x, pos_y)
	ctx.props[i].r = optim_color[0]
	ctx.props[i].g = optim_color[1]
	ctx.props[i].b = optim_color[2]
	ctx.improvements[i] = glyph_improvement(curr, target, ctx.props[i])

	free_all(context.temp_allocator)
}

context_init :: proc(target: ^Image_Data, curr: ^Image_Data, cache: Multiscale_SDF_Cache,
					proposal_count: int, R: f32) -> ^Iter_Ctx {
	ctx := new(Iter_Ctx)
	ctx.target = target
	ctx.curr = curr
	ctx.w = target.w
	ctx.h = target.h
	ctx.resid = make_resid_image(curr, target)
	ctx.sq_resid = make_sq_residual_image(curr, target)
	ctx.cache = cache
	ctx.R = R
	ctx.proposal_count = proposal_count
	ctx.allocator = context.allocator
	ctx.props = make([]Glyph, proposal_count, context.allocator)
	ctx.improvements = make([]f32, proposal_count, context.allocator)
	return ctx
}
context_deinit :: proc(ctx: ^Iter_Ctx) {
	delete(ctx.resid.data)
	delete(ctx.sq_resid.data)
	delete(ctx.props)
	delete(ctx.improvements)
	free(ctx)
}

main_algorithm_par :: proc(target: ^Image_Data, final_iters: int, proposal_count: int,
						cache: Multiscale_SDF_Cache, save_path:string, start_path:string,
						num_threads: int, iters_done: int=0, save_iters: int=20) {
	pool := pool_init(num_threads, cache)
	defer pool_deinit(pool)

	iters_done_f32 := f32(iters_done)
	R := 10.0 + math.sqrt_f32(iters_done_f32)
	R = 80.0 if R > 80.0 else R
	curr: ^Image_Data
	non_improve := 0
	if start_path == "" {
		//no starting point; start from scratch
		curr = init_empty_rgba(target.w,target.h)
	} else {
		//start from the path
		ok_start: bool
		curr, ok_start = image_from_path(start_path)
		ensure(ok_start)
	}
	ctx := context_init(target, curr, cache, proposal_count, R)
	defer context_deinit(ctx)
	pool.ctx = ctx
	iters := 0
	for iters<final_iters {
		//update residual images
		update_resid_image(ctx.curr, ctx.target, &ctx.resid)
		update_sq_resid_image(ctx.curr, ctx.target, &ctx.sq_resid)
		//make saliency table
		saliency := blur(ctx.resid.data, ctx.resid.w, ctx.resid.h) 
		normalize_vals(saliency.data)
		ctx.saliency_tbl = init_alias_table(saliency.data)

		sync.lock(&pool.mu)
    	pool.ctx = ctx
    	pool.next_job = 0
    	pool.done = 0
    	pool.generation += 1
		sync.cond_broadcast(&pool.cv)

		//wait for work to be done
		for pool.done < len(pool.threads) && !pool.stop {
		    sync.cond_wait(&pool.cv, &pool.mu)
		}
		sync.unlock(&pool.mu)

		//softmax(ctx.improvements, ctx.R, 10)
		//sample
		//normalize_vals(ctx.improvements)
		//tbl_g := init_alias_table(ctx.improvements)
		//glyph_ind := sample_from_tbl(tbl_g)	
		//deinit_alias_table(tbl_g)

		//paint the glyph we selected *if it actually improves error*
		//if ctx.improvements[glyph_ind] > 0 {
		//	ok_paint := paint_glyph(ctx.curr, ctx.props[glyph_ind])
		//	ensure(ok_paint)
		//}

		//select the best glyph
		best_ind := -1
		best_improvement: f32 = 0
		for i in 0..<len(ctx.improvements) {
    		if ctx.improvements[i] > best_improvement {
        		best_improvement = ctx.improvements[i]
        		best_ind = i
    		}
		}

		if best_ind == -1 {
			//if no improvement, don't paint
    		fmt.println("No improving proposal found.")
			non_improve += 1
		} else {
			ok_paint := paint_glyph(ctx.curr, ctx.props[best_ind])
			ensure(ok_paint)
		}

		fmt.println("Iteration complete. Number of iterations done: ", iters,
    	        " improvement: ", best_improvement, " number of non-improving iterations: " , non_improve)	
		iters += 1
		if iters % save_iters == 0 {
			//save

			//write the current iteration as a string
			buf: [8]byte
			iters_str := strconv.write_int(buf[:], i64(iters_done+iters), 10)
			strings_to_combine := [4]string{save_path, "/iter", iters_str, ".png" }
			save_file, err := strings.concatenate(strings_to_combine[:])
			ensure(err == nil)
			
			ok_save := save_png(save_file, ctx.curr)
			ensure(ok_save)
			delete(save_file)

		}

		delete(saliency.data)
		deinit_alias_table(ctx.saliency_tbl)
		if ctx.R >= 80.0 do continue
		ctx.R = 10.0 + math.sqrt_f32(iters_done_f32 + f32(iters))
	}
}

sample_rotation :: proc() -> f32 {
	//uniformly sample an angle
	return rand.float32_range(0, 2*math.PI)
}

blur :: proc(arr: []int, w,h:int, range: int = 1) -> One_Channel_Img_f32 {
	//blurs the []u8 slice,
	//interpreted as an image of width w and height h.
	//has to allocate.
	assert(len(arr) == w * h)
	out := make([]f32, w * h)
	for y in 0..<h {
		for x in 0..<w {
			//x,y is the center pixel.
			//blur using a square around the center,
			//with "radius" = range
			x_min := 0 if x - range < 0 else x - range
			x_max := w if x + range + 1 > w else x + range + 1
			y_min := 0 if y - range < 0 else y - range
			y_max := h if y + range + 1 > h else y + range + 1
			px: f32 = 0
			squares: f32 = (f32(2) * f32(range)+f32(1)) * (f32(2) * f32(range)+f32(1))
			weight := f32(1)/squares //uniform weight
			for suby in y_min..<y_max {
				for subx in x_min..<x_max {
					px += weight * f32(arr[suby * w + subx])
				}
			}
			out[y * w + x] = px
		}
	}
	return One_Channel_Img_f32{w = w, h = h, data = out}
}

Position :: struct {
	x, y: int
}
glyph_improvement :: proc(curr, target: ^Image_Data, g: Glyph) -> f32 {

	//compute how much *better* the glyph makes the residual error
	assert(curr.w == target.w)
	assert(curr.h == target.h)
	gR := f32(g.r)
	gG := f32(g.g)
	gB := f32(g.b)
	before: f32 = 0
	after: f32 = 0

	x0 := g.x - g.w/2
	y0 := g.y-g.h/2

	for mask_y in 0..<g.h {
		dst_y := y0 + mask_y
		if dst_y < 0 || dst_y >= curr.h do continue //out of img bound
		for mask_x in 0..<g.w {
			dst_x := x0 + mask_x
			if dst_x < 0 || dst_x >= curr.w do continue 

			m_alpha := g.bmap[mask_y * g.w + mask_x]
			if m_alpha == 0 do continue
			m_alpha_f32 := f32(m_alpha)/255.0 //[0,1] easier for math
			cur := curr.pixels[dst_y * curr.w + dst_x]
			tar := target.pixels[dst_y * curr.w + dst_x]
			//get pixels
			curR := f32(cur[0]); curG := f32(cur[1]); curB := f32(cur[2])
			tarR := f32(tar[0]); tarG := f32(tar[1]); tarB := f32(tar[2])

			//error before
			dR := tarR - curR
			dG := tarG - curG
			dB := tarB - curB
			before += dR*dR + dG*dG + dB*dB //sum sq

			//blend to get new px
			inv: f32 = 1.0 - m_alpha_f32
			newR := inv * curR + m_alpha_f32 * gR
			newG := inv * curG + m_alpha_f32 * gG
			newB := inv * curB + m_alpha_f32 * gB
			//error after
			eR := tarR - newR
			eG := tarG - newG
			eB := tarB - newB
			after += eR*eR + eG*eG + eB*eB
		}
	}
	return before-after
}
softmax :: proc(arr: []f32, r: f32 = 10, k: int = 10) {
	assert(k < len(arr))
	//softmaxes the array arr in place, using odds ratio r
	//between the best and k-th worst elements

	//has to sort; beware of complexity

	cpy := make([]f32, len(arr))
	defer delete(cpy)
	mem.copy(&cpy[0], &arr[0], len(arr) * 4) //4 bytes per element
	slice.sort(cpy[:]) //sort

	best := cpy[len(cpy)-1]
	kworst := cpy[k-1]
	lambda: f32 = math.ln_f32(f32(r))/(best-kworst+0.00001) //odds ratio between best/worst is r
	//now apply softmax
	for w in 0..<len(arr) {
		arr[w] = math.exp_f32(lambda * (arr[w]-best))
	}
}

optimal_color :: proc(curr, target: ^Image_Data, bmap: Glyph_Bitmap, x,y: int) -> [3]u8 {
	//computes optimal color for a glyph
	//at a position in the target image
	//(solution found with calculus)
	assert(curr.w == target.w)
	assert(curr.h == target.h)
	den: f32 = 0
	numR: f32 = 0
	numG: f32 = 0
	numB: f32 = 0
	x0 := x - bmap.w/2
	y0 := y-bmap.h/2
	for mask_y in 0..<bmap.h {
		dst_y := y0 + mask_y
		if dst_y < 0 || dst_y >= curr.h do continue //out of img bound
		for mask_x in 0..<bmap.w {
			dst_x := x0 + mask_x
			if dst_x < 0 || dst_x >= curr.w do continue 

			m_alpha := bmap.mask[mask_y * bmap.w + mask_x]
			if m_alpha == 0 do continue
			m_alpha_f32 := f32(m_alpha)/255.0
			m_alpha_sq := m_alpha_f32 * m_alpha_f32
			den += m_alpha_sq
			
			dst_index := dst_y * curr.w + dst_x
			currR := f32(curr.pixels[dst_index][0])
			currG := f32(curr.pixels[dst_index][1])
			currB := f32(curr.pixels[dst_index][2])
			targR := f32(target.pixels[dst_index][0])
			targG := f32(target.pixels[dst_index][1])
			targB := f32(target.pixels[dst_index][2])

			//numerator term: m_alpha * targ + (m_alpha_sq-m_alpha)*curr
			k := m_alpha_sq - m_alpha_f32
			numR += m_alpha_f32 * targR + k * currR
			numG += m_alpha_f32 * targG + k * currG
			numB += m_alpha_f32 * targB + k*currB
		}
	}
	cR := numR/den
	cG := numG/den
	cB := numB/den

	//clamp to [0,255]
	cR = 0 if cR < 0 else cR
	cR = 255 if cR > 255 else cR

	cG = 0 if cG < 0 else cG
	cG = 255 if cG > 255 else cG

	cB = 0 if cB < 0 else cB
	cB = 255 if cB > 255 else cB

	//now round
	cR_u8: u8 = u8(int(cR+0.5))
	cG_u8: u8 = u8(int(cG+0.5))
	cB_u8: u8 = u8(int(cB+0.5))
	return [3]u8{cR_u8, cG_u8, cB_u8}
}

glyph_score :: proc(resid: One_Channel_Img, bmap: Glyph_Bitmap, x,y:int) -> f32 {
	//inner product of residuals and bitmap squared
	//normalized by the sum of the squares of the bitmap
	//(basically, how much the glyph mass agrees with the residual)
	inn_prod := 0
	x0 := x - bmap.w/2
	y0 := y - bmap.h/2
	sum_sq_mask := 0

	for mask_y in 0..<bmap.h {
		dst_y := y0 + mask_y
		if dst_y < 0 || dst_y >= resid.h do continue //out of img bound
		
		for mask_x in 0..<bmap.w {
			dst_x := x0 + mask_x
			if dst_x < 0 || dst_x >= resid.w do continue 

			m_alpha := bmap.mask[mask_y * bmap.w + mask_x]
			if m_alpha == 0 do continue
			
			inn_prod += int(m_alpha) * resid.data[dst_y * resid.w + dst_x]	
			sum_sq_mask += int(m_alpha) * int(m_alpha)
		}
	}
	inn_prod = inn_prod * inn_prod
	return f32(inn_prod)/f32(sum_sq_mask)
}



energy_density_frame :: proc(energy: One_Channel_Img, f: Frame) -> f32 {
	//sum of the squared residuals
	//divided by the number of pixels.
	//a simple heuristic to decide what glyph scales are "good"
	total_resid := sq_residual_frame(energy, f)
	return f32(total_resid)/f32(f.w * f.h)	
}

frame_from_glyph :: proc(g: Glyph) -> Frame {
	return Frame{x0 = g.x - g.w/2, y0 = g.y - g.h/2, w = g.w, h = g.h}
}

frame_from_dimensions :: proc(x,y: int, d: Dimensions) -> Frame {
	return Frame{x0 = x-d.w/2, y0 = y-d.h/2, w = d.w, h = d.h}
}

sq_residual_frame :: proc(energy: One_Channel_Img, fr: Frame) -> int {
	w := energy.w
	h := energy.h

	//Computes the residual
	//pixel-wise within the specified frame (rectangular region).
	res := 0
	//we only care about the region that actually overlaps with the image.
	min_y := 0 if fr.y0 < 0 else fr.y0
	max_y := h if fr.y0+fr.h > h else fr.y0 + fr.h  //exclusive btw
	min_x := 0 if fr.x0 < 0 else fr.x0
	max_x := w if fr.x0 + fr.w > w else fr.x0 + fr.w

	for y in min_y..<max_y {
		for x in min_x..<max_x {
			res += energy.data[y*w+x]
		}
	}
	return res
}

sq_residual_px :: proc(approx: [4]u8, target: [4]u8) -> int {
	res := 0
	for i in 0..<3 {
		res += ((int(target[i]) - int(approx[i])) * (int(target[i]) - int(approx[i]))) * int(target[3]) //weigh by alpha of target
	}
	return res
}
make_sq_residual_image :: proc(approx, target: ^Image_Data) -> One_Channel_Img {
	assert(approx.w == target.w, "Incompatible image widths")
	assert(approx.h == target.h, "Incompatible image widths")
	w := approx.w
	h := approx.h

	out := make([]int, w * h)
	for y in 0..<h {
		for x in 0..<w {
			out[y * w + x] = sq_residual_px(approx.pixels[y*w+x], target.pixels[y*w+x])
		}
	}
	return One_Channel_Img{w=w,h=h,data=out}
}
update_sq_resid_image :: proc(approx, target: ^Image_Data, resid: ^One_Channel_Img) {
	w := approx.w
	h := approx.h
	for y in 0..<h {
		for x in 0..<w {
			resid.data[y * w + x] = sq_residual_px(approx.pixels[y*w+x], target.pixels[y*w+x])
		}
	}
}
make_resid_image :: proc(approx, target: ^Image_Data) -> One_Channel_Img {
	
	assert(approx.w == target.w, "Incompatible image widths")
	assert(approx.h == target.h, "Incompatible image widths")
	w := approx.w
	h := approx.h

	out := make([]int, w*h)
	for y in 0..<h {
		for x in 0..<w {
			out[y * w + x] = resid_px(approx.pixels[y*w+x], target.pixels[y*w+x])
		}
	}
	return One_Channel_Img{w=w,h=h,data=out}
}
update_resid_image :: proc(approx, target: ^Image_Data, resid: ^One_Channel_Img) {
	w := approx.w
	h := approx.h
	for y in 0..<h {
		for x in 0..<w {
			resid.data[y * w + x] = resid_px(approx.pixels[y*w+x], target.pixels[y*w+x])
		}
	}
}
resid_px :: proc(approx: [4]u8, target: [4]u8) -> int {
	res := 0
	for i in 0..<3 {
		res += ((int(target[i]) - int(approx[i])) *(int(target[i])-int(approx[i]))) * int(target[3])
	}
	return int(math.sqrt_f32(f32(res))+0.5) //this is the only difference, im just lazy
}



