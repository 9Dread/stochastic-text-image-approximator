package text_imgapproximator

import "core:os"
import "core:strings"
import "core:fmt"

Arg_Kind :: enum {
	Flag,
	Option,
}

Arg :: struct {
	kind: Arg_Kind,
	name: string,
	value: string,
}

parse_args :: proc(args: []string, valid_names: []string) -> (parsed: []Arg, ok: bool) {
	//pre-alloc output, can't be longer than len(args)
	out := make([]Arg, len(args)) 

	i := 0
	j := 0

	for i < len(args) {
		a := args[i]
		if strings.has_prefix(a, "--") {
			//check validity of names
			if !check_valid_name(a[2:], valid_names) {
				fmt.println("Error - Unrecognized argument: ", a)
				return out, false
			}
			//if arg is an option, should have
			//another argument in the array
			//and it should not start with --
			if i + 1 < len(args) && !(strings.has_prefix(args[i+1], "--")) {
				out[j] = Arg{ kind = .Option, name = a[2:], value = args[i+1] }
				j += 1
				i += 2
				continue
			} else {
				//flag
				out[j] = Arg{ kind = .Flag, name = a[2:]}
				j += 1
				i += 1
				continue
			}
		}
		i += 1
	}
	return out, true
}

check_valid_name :: proc(name: string, valid_names: []string) -> bool {
	for valid in valid_names {
		if name == valid do return true
	}
	return false
}

Args_Map :: struct {
	flags: map[string]bool,
	opts: map[string]string,
}

make_args_map :: proc(parsed: []Arg) -> Args_Map {
	out := Args_Map{
		flags = make(map[string]bool),
		opts = make(map[string]string)
	}

	for arg in parsed {
		switch arg.kind {
		case .Option:
			out.opts[arg.name] = arg.value
		case .Flag:
			out.flags[arg.name] = true
		}
	}
	return out
}

delete_args_map :: proc(m: Args_Map) {
	delete(m.flags)
	delete(m.opts)
}

map_from_args_arr :: proc(args: []string, valid_names: []string) -> (result: Args_Map, ok: bool) {
	//abstracts away calling parse_args before making a map,
	//which is confusing since the []Arg array may be immediately freed
	//after making the map (it does not own the data; os.args does)

	parsed, ok_parse := parse_args(args, valid_names)
	defer delete(parsed)
	if !ok_parse {
		return Args_Map{}, false
	}
	amap := make_args_map(parsed)
	return amap, true
}
