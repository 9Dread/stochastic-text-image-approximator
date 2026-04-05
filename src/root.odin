package text_imgapproximator

import "core:fmt"
import "core:image"
import "core:os"
import "core:image/png"
import vmem "core:mem/virtual"
import "core:strconv"

main :: proc() {
	valid_args := [?]string{ "out", "in", "ttf", "start", "iters_done", "threads", "dict", "save_iters", "iterations", "proposal_count" }
	args, ok_args := map_from_args_arr(os.args, valid_args[:])
	if !ok_args do return

	img, ok_img := image_from_path(args.opts["in"])
	if !ok_img do return

	px_heights := make_pxh_scale(max = f32(img.h)/2, min=12)
	//initialize dictionary of codepoints
	dict := args.opts["dict"]
	if dict == "" {
		fmt.println("Please provide a dictionary of comma-separated characters with the --dict option.")
		return
	}
	num_chars := len(dict)/2 + 1
	i := 1
	for i < len(dict) {
		//all odd entries should be commas
		if dict[i] != ',' {
			fmt.println("Incorrectly-formatted character dictionary supplied with --dict.")
			fmt.println("This should be supplied as char1,char2,...")
			fmt.println("e.g. A,B,C,1,2,3")
			return
		}
		i += 2
	}
	codepoints := make([]rune, num_chars)
	i = 0
	j := 0
	for i < len(dict) {
		codepoints[j] = cast(rune)dict[i]
		i += 2
		j += 1
	}
	cache, ok_cache := init_from_ttf(args.opts["ttf"], px_heights, codepoints)
	if !ok_cache do return
	defer deinit_multiscale_cache(cache)
	iters_done: int
	if args.opts["iters_done"] == "" {
		iters_done = 0
	} else {
		ok_iters: bool
		iters_done, ok_iters = strconv.parse_int(args.opts["iters_done"])
		if !ok_iters {
			fmt.println("Parsing iters_done failed.")
			return
		}
		ensure(iters_done >= 0)
	}
	threads: int
	if args.opts["threads"] == "" {
		threads = 1
	} else {
		ok_threads: bool
		threads, ok_threads = strconv.parse_int(args.opts["threads"])
		if !ok_threads {
			fmt.println("Parsing threads failed.")
			return
		}
		ensure(threads > 0)
	}
	save_iters: int
	if args.opts["save_iters"] == "" {
		save_iters = 50
	} else {
		ok_save_iters: bool
		save_iters, ok_save_iters = strconv.parse_int(args.opts["save_iters"])
		if !ok_save_iters {
			fmt.println("Parsing save_iters failed.")
			return
		}
		ensure(save_iters > 0)
	}
	iters: int
	if args.opts["iterations"] == "" {
		fmt.println("Please supply the number of iterations to complete with the --iterations option.")
		return
	} else {
		ok_iters: bool
		iters, ok_iters = strconv.parse_int(args.opts["iterations"])
		if !ok_iters {
			fmt.println("Parsing iterations failed.")
			return
		}
		ensure(iters > 0)
	}
	proposal_count: int
	if args.opts["proposal_count"] == "" {
		proposal_count = 64
	} else {
		ok_props: bool
		proposal_count, ok_props = strconv.parse_int(args.opts["proposal_count"])
		if !ok_props {
			fmt.println("Parsing proposal_count failed.")
			return
		}
		ensure(proposal_count > 0)
	}

	main_algorithm_par(target = img, final_iters = iters, proposal_count = proposal_count, cache = cache, 
					save_path=args.opts["out"], start_path=args.opts["start"], num_threads=threads, 
					iters_done = iters_done, save_iters = save_iters)

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


