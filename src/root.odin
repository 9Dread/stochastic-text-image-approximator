package text_imgapproximator

import "core:fmt"
import "core:image"
import "core:os"
import "core:image/png"
import vmem "core:mem/virtual"
import "core:strconv"

main :: proc() {
	valid_args := [?]string{ "out", "in", "ttf", "start", "iters_done" }
	//out, in, ttf are required.
	//if start is given, it's
	//interpreted as a path to the starting image (won't init from scratch).

	args, ok_args := map_from_args_arr(os.args, valid_args[:])
	if !ok_args do return

	//img := init_empty_rgba(600, 600)
	//defer image_delete(img)

	img, ok_img := image_from_path(args.opts["in"])
	if !ok_img do return
	//img := init_empty_rgba(1920, 1080)
	//defer image_delete(img)

	//cache sdfs


	px_heights := make_pxh_scale(max = f32(img.h)/2, min=12)
	codepoints := make([]rune, 2)
	codepoints[0] = '6'
	codepoints[1] = '7'
	cache, ok_cache := init_from_ttf(args.opts["ttf"], px_heights, codepoints)
	if !ok_cache do return
	defer deinit_multiscale_cache(cache)
	iters_done, ok_iters := strconv.parse_int(args.opts["iters_done"])
	if !ok_iters {
		fmt.println("Parsing iters_done failed.")
		return
	}
	main_algorithm(target = img, final_glyphs = 1000, proposal_count = 32, cache = cache, 
					save_path=args.opts["out"], start_path=args.opts["start"], iters_done = iters_done)

	image_delete(img)
	delete_args_map(args)
}

make_pxh_scale :: proc(min: f32, max: f32) -> []f32 {
	count := 0
	height := max
	for height >= min {
		count += 1
		height = height/2
	}

	//second pass to actually put in the heights
	out := make([]f32, count)
	i := 0
	height = max
	for height >= min {
		out[i] = height
		height = height/2
		i = i + 1
	}
	return out
}


