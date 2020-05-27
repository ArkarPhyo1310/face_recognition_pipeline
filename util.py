from PIL import ImageDraw, ImageFont


def draw_bb_on_img(faces, img):
    draw = ImageDraw.Draw(img)
    fs = max(20, round(img.size[0] * img.size[1] * 0.000005))
    font = ImageFont.truetype('font/Raleway.ttf', fs)
    margin = 5

    for face in faces:
        text = "%s %.2f%%" % (face.top_prediction.label.upper(), face.top_prediction.confidence * 100)
        text_size = font.getsize(text)
        color = "green"
        if "UNKNOWN" in text:
            color = "red"
        # bounding box
        draw.rectangle(
            (
                (int(face.bb.left), int(face.bb.top)),
                (int(face.bb.right), int(face.bb.bottom))
            ),
            outline=color,
            width=4
        )

        # text background
        draw.rectangle(
            (
                (int(face.bb.left - margin), int(face.bb.bottom) + margin),
                (int(face.bb.left + text_size[0] + margin), int(face.bb.bottom) + text_size[1] + 3 * margin)
            ),
            fill='black'
        )

        # text
        draw.text(
            (int(face.bb.left), int(face.bb.bottom) + 2 * margin),
            text,
            font=font
        )


def draw_fps(frame, text, vidw):
    draw = ImageDraw.Draw(frame)
    font_size = max(15, round(frame.size[0] * frame.size[1] * 0.000005))
    font = ImageFont.truetype('font/Raleway.ttf', font_size)
    margin = 4
    draw.rectangle(
        [
            (vidw - 75, 0),
            (vidw, 25)
        ],
        fill='black'
    )

    draw.text(xy=(vidw - 65 + margin, margin), text=text, fill='white', font=font)


def draw_frame_count(frame, text):
    draw = ImageDraw.Draw(frame)
    font_size = max(15, round(frame.size[0] * frame.size[1] * 0.000005))
    font = ImageFont.truetype('font/Raleway.ttf', font_size)
    margin = 4
    draw.rectangle(
        [
            (0, 0),
            (50, 25)
        ],
        fill='black'
    )

    draw.text(xy=(margin, margin), text=text, fill='white', font=font)