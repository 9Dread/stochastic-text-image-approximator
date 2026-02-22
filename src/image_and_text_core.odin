package text_imgapproximator

import stbi "vendor:stb/image"
import "core:image"
import "core:image/png"
import "core:image/jpeg"
import "core:fmt"
import "core:mem"
import "core:strings"
import "core:slice"

//utils for image processing + 
//processing glyphs into rasters.

RGBA_px :: [4]u8
Image_Data :: struct {
	//creating from file yields an ^image.Image.
	//This is rather annoying and abstracted away.
	w, h: int,
	pixels: []RGBA_px
}

image_delete :: proc(img: ^Image_Data) {
	delete(img.pixels)
	free(img)
}

image_from_path :: proc(path: string) -> (result: ^Image_Data, ok: bool) {
	
	img, err := image.load_from_file(path)	
	defer image.destroy(img)

	if err != nil {
		fmt.println("Failed to read image")
		fmt.println(err)
		return nil, false
	}

	ok_add_alpha := image.alpha_add_if_missing(img)

	if !ok_add_alpha {
		fmt.println("Failed to add alpha channel to image")
		return nil, false
	}

	//clone data into 
	//our own slice
	w := img.width
	h := img.height
	raw_data := make([]RGBA_px, w * h)
	for i := 0; i < w * h; i += 1 {
		for j := i * 4; j < i * 4 + 4; j += 1 {
			raw_data[i][j-(i*4)] = img.pixels.buf[j]
		}
	}
	out := new(Image_Data)
	out.w = w
	out.h = h
	out.pixels = raw_data
	return out, true
}

init_empty_rgba :: proc(w: int, h: int) -> ^Image_Data {
	data := make([]RGBA_px, w * h)
	for i := 0; i < w * h; i += 1 {
		data[i][3] = 255
	}
	out := new(Image_Data)
	out.w = w
	out.h = h
	out.pixels = data
	return out
}

save_png :: proc(path: string, img_data: ^Image_Data) -> bool {

	//using stb/image as backend for png saving
	//since core:image doesn't offer this
	//(unless im stupid--real possibility btw)

	w := i32(img_data.w)
	h := i32(img_data.h)
	c := i32(4) //should always be rgba

    stride := w * c

	path_cstring := strings.clone_to_cstring(path, context.temp_allocator)
	defer free_all(context.temp_allocator)
	ok: i32
	ok = stbi.write_png(path_cstring, w, h, c, mem.raw_data(img_data.pixels), stride)
    if ok == 0 {
        fmt.println("failed to write png")
        return false
    }

    return true
}

Glyph :: struct {
	x: int,
	y: int,
	w: int,
	h: int,
	bmap: []u8,
	r: u8,
	g: u8,
	b: u8
}

glyph_from_bmap :: proc(bitmap: Glyph_Bitmap, x,y: int, r,g,b: u8) -> Glyph {
	return Glyph{
		x = x,
		y = y,
		w = bitmap.w,
		h = bitmap.h,
		bmap = bitmap.mask,
		r = r,
		g = g,
		b = b
	}
}

paint_glyph :: proc(img: ^Image_Data, glyph: Glyph) -> bool {

	//we interpret x/y as a center point. So
	//x0, y0 (top left) are computed

	x0 := glyph.x - glyph.w / 2
	y0 := glyph.y - glyph.h / 2
	for mask_y in 0..<glyph.h {
		dst_y := y0 + mask_y
		if dst_y < 0 || dst_y >= img.h do continue //out of img bound
		
		for mask_x in 0..<glyph.w {
			dst_x := x0 + mask_x
			if dst_x < 0 || dst_x >= img.w do continue 

			m_alpha := glyph.bmap[mask_y * glyph.w + mask_x]
			if m_alpha == 0 do continue //nothing to overlay

			m_px := [4]u8{glyph.r, glyph.g, glyph.b, m_alpha}
			orig_px := &img.pixels[dst_y * img.w + dst_x]
			ok := paint_px(orig_px[:], m_px[:])
			if !ok do return false
		}
	}
	return true
}

paint_px :: proc(dst: []u8, over: []u8) -> bool {
    if len(dst) != 4 || len(over) != 4 do return false

    over_r := int(over[0])
    over_g := int(over[1])
    over_b := int(over[2])
    a := int(over[3])

    dst_r := int(dst[0])
    dst_g := int(dst[1])
    dst_b := int(dst[2])
    dst_a := int(dst[3])

    inv := 255 - a

    out_a := a + (dst_a * inv + 127) / 255
    out_r := (over_r * a + dst_r * inv + 127) / 255
    out_g := (over_g * a + dst_g * inv + 127) / 255
    out_b := (over_b * a + dst_b * inv + 127) / 255

    dst[0] = u8(out_r)
    dst[1] = u8(out_g)
    dst[2] = u8(out_b)
    dst[3] = u8(out_a)
    return true
}


