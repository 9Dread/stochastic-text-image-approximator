package text_imgapproximator

//utils for rendering all necessary glyphs from a font to
//bitmaps so that they may be used for computation.
//abstracts away dealing with fonts for the rest of
//the project.

import "core:fmt"
import "core:os"
import "core:mem"
import stbtt "vendor:stb/truetype"
import "core:math"
import "base:runtime"

//we need to arbitrarily rotate
//glyphs; SDFs help with this
Glyph_SDF :: struct {
	w, h: int,
	onedge: f32,
	dist_scale: f32,
	sdf: []u8
}

Glyph_Bitmap :: struct {
	w, h: int,
	mask: []u8
}

Glyph_Cache :: struct {
	font: ^stbtt.fontinfo,
	ttf_data: []u8,
	px_h: f32,
	scale: f32,
	glyphs: map[rune]Glyph_SDF,
}


make_glyph_sdf :: proc(font: ^stbtt.fontinfo, scale: f32, glyph: int,
                       padding: int = 8,
                       onedge_value: u8 = 128,
                       pixel_dist_scale: f32 = 16.0) -> (sdf: Glyph_SDF, ok: bool) {
  	w, h, xoff, yoff: i32

  	//returns ^u8 (C pointer) or nil on failure
  	ptr := stbtt.GetGlyphSDF(font, scale, i32(glyph), i32(padding), onedge_value,
                           pixel_dist_scale, &w, &h, &xoff, &yoff)
  	if ptr == nil || w <= 0 || h <= 0 {
		fmt.println("stbtt.GetGlyphSDF failed")
  		return Glyph_SDF{}, false
  	}

  	//copy the sdf into our own memory
  	out := make([]u8, int(w) * int(h))
  	mem.copy(&out[0], ptr, len(out)) //use ptr to first element in slice

	//free the stb allocated sdf
  	stbtt.FreeSDF(ptr, font.userdata)

  	return Glyph_SDF{w=int(w), h=int(h), sdf=out, onedge=f32(onedge_value), dist_scale=pixel_dist_scale}, true
}

smoothstep :: proc(a,b,x:f32) -> f32 {
	t := (x-a)/(b-a)
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return t * t * (3-2*t)
}

sample_sdf_bilinear :: proc(g: Glyph_SDF, u, v: f32) -> f32 {
    x0 := int(u)
	y0 := int(v)
    x1 := x0 + 1
	y1 := y0 + 1
    if x0 < 0 || y0 < 0 || x1 >= g.w || y1 >= g.h {
        return 0 //outside; very negative distance effectively, so 0 alpha
    }

    fx := u - f32(x0)
    fy := v - f32(y0)

    i00 := g.sdf[y0*g.w + x0]
    i10 := g.sdf[y0*g.w + x1]
    i01 := g.sdf[y1*g.w + x0]
    i11 := g.sdf[y1*g.w + x1]

    a := f32(i00)*(1-fx) + f32(i10)*fx
    b := f32(i01)*(1-fx) + f32(i11)*fx
    return a*(1-fy) + b*fy
}

make_rotated_from_sdf :: proc(g: Glyph_SDF, theta: f32, out_scale: f32 = 1.0, allocator: runtime.Allocator) -> Glyph_Bitmap {
    c := f32(math.cos_f32(theta))
    s := f32(math.sin_f32(theta))
	
    in_ax := f32(g.w) * 0.5
    in_ay := f32(g.h) * 0.5

    //compute output bounds that fit the rotated rectangle.
    //half-extents of input rect in its local space:
    hx := f32(g.w) * 0.5 * out_scale
    hy := f32(g.h) * 0.5 * out_scale

    //rotated half extents (AABB of rotated rectangle):
    out_hx := abs(c*hx) + abs(s*hy)
    out_hy := abs(s*hx) + abs(c*hy)

    out_w := int(math.ceil_f32(2*out_hx)) + 2
    out_h := int(math.ceil_f32(2*out_hy)) + 2

	out_ax := f32(out_w) * 0.5
	out_ay := f32(out_h) * 0.5

    out := Glyph_Bitmap{
        w = out_w,
        h = out_h,
        mask = make([]u8, out_w*out_h, allocator)
    }

    //antialias half-width in distance units; if output is scaled, shrink accordingly.
    //this makes the transition about ~1 output pixel.
    w := 0.5 / out_scale

    for y in 0..<out_h {
        for x in 0..<out_w {
            //output pixel relative to anchor
            dx := (f32(x) + 0.5) - out_ax
            dy := (f32(y) + 0.5) - out_ay

            //map back into input SDF coordinates;
            //inverse rotation + inverse scale, then re-center at input anchor
            u := ( c*dx + s*dy )/out_scale + in_ax
            v := (-s*dx + c*dy )/out_scale + in_ay

            if u < 0 || v < 0 || u >= f32(g.w-1) || v >= f32(g.h-1) {
                continue
            }

            sdfv := sample_sdf_bilinear(g, u, v)
            dist := (sdfv - g.onedge) / g.dist_scale

            a := smoothstep(-w, +w, dist) // 0..1
            out.mask[y*out_w + x] = u8(clamp(a*255.0, 0, 255))
        }
    }

    return out
}

deinit_cache :: proc(cache: Glyph_Cache) {
	sdf: Glyph_SDF

	//delete all sdfs
	for key in cache.glyphs {
		sdf = cache.glyphs[key]
		delete(sdf.sdf)
	}
	
	//delete the map 
	delete(cache.glyphs)
}

get_codepoint_SDF :: proc(cache: ^Glyph_Cache, codepoint: rune) -> (bmap: Glyph_SDF, ok: bool) {
	if g, ok := cache.glyphs[codepoint]; ok {
		return g, true //already cached
	}

	glyph := stbtt.FindGlyphIndex(cache.font, codepoint)
	sdf, ok_sdf := make_glyph_sdf(cache.font, cache.scale, int(glyph))
	if !ok_sdf {
		return Glyph_SDF{}, false
	}

	cache.glyphs[codepoint] = sdf
	return sdf, true
}

cache_codepoints :: proc(cache: ^Glyph_Cache, codepoints: []rune) -> bool {
	for r in codepoints {
		sdf, ok := get_codepoint_SDF(cache, r)
		if !ok do return false
	}
	return true
}

//We will discretize scales,
//so we need a map of caches for each scale.
Multiscale_SDF_Cache :: struct {
	ttf_data: []u8,
	font: ^stbtt.fontinfo,
	px_heights: []f32,
	valid_codepoints: []rune,
	scales_cache: map[int]Glyph_Cache,
	size_frames: map[int]Dimensions //an approximation of bounding frames for each px_height
}
Dimensions :: struct {
	w,h: int
}

init_from_ttf :: proc(path: string, px_heights: []f32, valid_codepoints: []rune) -> (out: Multiscale_SDF_Cache, ok: bool) {
	cache: Multiscale_SDF_Cache

	ttf_data, ttf_ok := os.read_entire_file(path)
	if !ttf_ok {
		fmt.println("Error - Failed to read file ", path)
		return Multiscale_SDF_Cache{}, false
	}	
	finfo := new(stbtt.fontinfo)

	if !stbtt.InitFont(finfo, &ttf_data[0], 0) {
		fmt.println("Error - stbtt.InitFont failed")
		return Multiscale_SDF_Cache{}, false
	}
	cache.font = finfo
	cache.ttf_data = ttf_data
	cache.px_heights = px_heights
	cache.valid_codepoints = valid_codepoints
	cache.scales_cache = make(map[int]Glyph_Cache)
	cache.size_frames = make(map[int]Dimensions)
	//for each scale, initialize the sdf cache
	for px_h in px_heights {
		sdf_cache := init_sdf_cache(cache.ttf_data, cache.font, px_h)
		//then cache all codepoints into it
		ok_cache := cache_codepoints(&sdf_cache, cache.valid_codepoints)
		key := int(px_h + 0.5)
		cache.scales_cache[key] = sdf_cache
		if !ok_cache do return Multiscale_SDF_Cache{}, false
	}

	//just use the first glyph
	glyph := cache.valid_codepoints[0]

	//For each size, cache a Dimension from the bitmap of glyph
	for height in cache.px_heights {

		bitmap := get_bitmap(cache, height, glyph, 0.0, 1.0, context.allocator)
		defer delete(bitmap.mask)
		cache.size_frames[int(height+0.5)] = Dimensions{w=bitmap.w,h=bitmap.h}
	}

	return cache, true
}

get_dimensions_from_cache :: proc(msc: Multiscale_SDF_Cache, px_h: f32) -> Dimensions {
	frame, ok_frame := msc.size_frames[int(px_h+0.5)]
	ensure(ok_frame)

	return frame
}

init_sdf_cache :: proc(ttf_data: []u8, font: ^stbtt.fontinfo, px_h: f32) -> Glyph_Cache {
	//assumes that ttf_data and font are initialized correctly
	cache: Glyph_Cache
	cache.font = font
	cache.ttf_data = ttf_data
	cache.px_h = px_h
	cache.scale = stbtt.ScaleForPixelHeight(cache.font, px_h)
	cache.glyphs = make(map[rune]Glyph_SDF)
	return cache
}

deinit_multiscale_cache :: proc(m: Multiscale_SDF_Cache) {
	sdf_cache: Glyph_Cache
	for key in m.scales_cache {
		sdf_cache = m.scales_cache[key]
		deinit_cache(sdf_cache)
	}
	delete(m.size_frames)
	delete(m.px_heights)
	delete(m.valid_codepoints)
	delete(m.ttf_data)
	delete(m.scales_cache)
	free(m.font)
}

//also need a way to index into a cache, so
get_SDF_from_cache :: proc(msc: Multiscale_SDF_Cache, px_h: f32, cp: rune) -> (res:Glyph_SDF, ok:bool) {
	//these should be in the cache already
	cache, ok_cache := msc.scales_cache[int(px_h+0.5)]
	if !ok_cache {
		fmt.println("Error - Attempt to get SDF from uncached pixel height")
		return Glyph_SDF{}, false
	}
	sdf, ok_sdf := cache.glyphs[cp]
	if !ok_sdf {
		fmt.println("Error - Attempt to get SDF from uncached codepoint")
		return Glyph_SDF{}, false
	}
	return sdf, true
}

//now we can get a bitmap from the cache
//at arbitrary rotation at any scale level
get_bitmap :: proc(msc: Multiscale_SDF_Cache, 
					px_h: f32, 
					cp: rune, 
					theta: f32, 
					out_scale: f32 = 1.0, 
					allocator: runtime.Allocator) -> Glyph_Bitmap {
	//index into cache then return the bitmap
	sdf, ok_sdf := get_SDF_from_cache(msc, px_h, cp)
	ensure(ok_sdf)
	return make_rotated_from_sdf(sdf,theta,out_scale, allocator) //beware: this allocates!
	//for our purposes, we will basically be freeing all bitmaps
	//at each iteration, so we will use an arena here.
}

