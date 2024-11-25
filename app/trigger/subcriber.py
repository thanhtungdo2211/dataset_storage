import os, json
import redis
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes
import asyncio
from io import BytesIO
from PIL import Image
import os
from PIL import Image, ImageDraw
import datetime


def get_current_time():
    now = datetime.datetime.now()
    current_time = now.strftime("Ngày %d, Tháng %m, Năm %Y, %H giờ %M phút")
    return current_time

def read_bboxes(file_path):
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes

def draw_bboxes(image, bboxes, color):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for bbox in bboxes:
        class_id, x_center, y_center, w, h = bbox
        x_min = (x_center - w / 2) * width
        y_min = (y_center - h / 2) * height
        x_max = (x_center + w / 2) * width
        y_max = (y_center + h / 2) * height
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        draw.text((x_min, y_min), str(class_id), fill=color)

def create_diff_image(bboxes_label, bboxes_predict, size):
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    width, height = size

    label_set = set(bboxes_label)
    predict_set = set(bboxes_predict)

    diff_bboxes = label_set.symmetric_difference(predict_set)

    for bbox in diff_bboxes:
        class_id, x_center, y_center, w, h = bbox
        x_min = (x_center - w / 2) * width
        y_min = (y_center - h / 2) * height
        x_max = (x_center + w / 2) * width
        y_max = (y_center + h / 2) * height
        draw.rectangle([x_min, y_min, x_max, y_max], outline='black', width=2)
        draw.text((x_min, y_min), str(class_id), fill='black')

    return image

def merge_images(image1, image2, image3):
    widths, heights = zip(*(i.size for i in [image1, image2, image3]))
    total_width = sum(widths)
    max_height = max(heights)
    
    new_image = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in [image1, image2, image3]:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return new_image

def process_and_merge_images(image_path, label_path, predict_path):
    image = Image.open(image_path).convert("RGB")
    size = image.size
    
    label_bboxes = read_bboxes(label_path)
    predict_bboxes = read_bboxes(predict_path)
    
    image_with_labels = image.copy()
    image_with_predicts = image.copy()
    
    draw_bboxes(image_with_labels, label_bboxes, color='red')
    draw_bboxes(image_with_predicts, predict_bboxes, color='blue')
    
    diff_image = create_diff_image(label_bboxes, predict_bboxes, size)
    
    merged_image = merge_images(image_with_labels, image_with_predicts, diff_image)
    
    return merged_image

click = True 

async def notify_difference(label_path, predict_path, image_path):
    global click
    click = False  
    
    t = get_current_time()
    keyboard = [
        [
            InlineKeyboardButton("🆙 Up to CVAT", callback_data=f'upToCvat+{predict_path.split("/")[-3]}_{t}'),
            InlineKeyboardButton("❌ Delete", callback_data=f'Delete+{predict_path.split("/")[-3]}_{t}')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    merged_image = process_and_merge_images(image_path, label_path, predict_path)
    image_byte_array = BytesIO()
    merged_image.save(image_byte_array, format='PNG')
    image_byte_array.seek(0)

    await app.bot.send_photo(chat_id=chat_id, photo=image_byte_array, caption=f"🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴\n⏰⏰⏰TIME : {t}\n📋📋📋TASK: {predict_path.split('/')[-3]}\n", reply_markup=reply_markup)
    

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global click
    query = update.callback_query
    await query.answer()
    choice = query.data.split('+')[0]
    task = query.data.split('+')[1]
    

    if choice == 'upToCvat':
        await query.edit_message_caption(caption=f"🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵\n⏰⏰⏰ TIME : {get_current_time()}\n📋📋📋 TASK : {task}\n⌛⌛⌛Processing")
        
    elif choice == 'Delete':
        await query.edit_message_caption(caption=f"🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵\n⏰⏰⏰ TIME : {get_current_time()}\n📋📋📋 TASK : {task}\n❌❌❌Cancelled")

    click = True 

async def listen_to_redis():
    pubsub = r.pubsub()
    pubsub.subscribe("file_differences")

    while True:
        if click:
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                data = json.loads(message['data'])
                print(data)
                predict_path = data['predict_path']
                label_path = data['label_path']
                image_path = data['image_path']
                await notify_difference(label_path, predict_path, image_path)
        await asyncio.sleep(1)  

def main():
    app.add_handler(CallbackQueryHandler(button))
    app.run_polling()

if __name__ == "__main__":
    app = ApplicationBuilder().token("6404179839:AAG4fD_BieNzlXvYEHevWRI8z1_dixJz1wU").build()
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    chat_id = "-4245611864"
    processed_files = set()
    loop = asyncio.get_event_loop()
    loop.create_task(listen_to_redis())
    main()
