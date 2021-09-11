from telebot import TeleBot, types
from pymongo import MongoClient
import os
from dsmodule import process_document

token = '1936856693:AAHuG4RV_AuELxRlnjsvOKucxJe3fy4nL5Q'
bot = TeleBot(token)
mdb = MongoClient('mongodb+srv://Demontego:574322538a@telebot.s1job.mongodb.net/tm_bot?authSource=admin&replicaSet=atlas-hwd1ci-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass&retryWrites=true&ssl=true').corruption
startKeyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
sendDocBtn = types.KeyboardButton('Отправить документ на проверку')
showDocBtn = types.KeyboardButton("Просмотреть отправленные документы")
startKeyboard.row(sendDocBtn)
startKeyboard.row(showDocBtn)

regions = ['Алтайский край',
 'Амурская область',
 'Архангельская область',
 'Астраханская область',
 'Белгородская область',
 'Брянская область',
 'Владимирская область',
 'Волгоградская область',
 'Вологодская область',
 'Воронежская область',
 'Еврейская автономная область',
 'Забайкальский край',
 'Ивановская область',
 'Кабардино-Балкарская Республика',
 'Калининградская область',
 'Калужская область',
 'Камчатский край',
 'Кемеровская область - Кузбасс',
 'Кировская область',
 'Костромская область',
 'Краснодарский край',
 'Красноярский край',
 'Курганская область',
 'Курская область',
 'Липецкая область',
 'Магаданская область',
 'Москва',
 'Московская область',
 'Мурманская область',
 'Ненецкий автономный округ',
 'Нижегородская область',
 'Новгородская область',
 'Новосибирская область',
 'Омская область',
 'Оренбургская область',
 'Орловская область',
 'Пермский край',
 'Приморский край',
 'Псковская область',
 'РСО-Алания',
 'Республика Адыгея  (Адыгея)',
 'Республика Алтай',
 'Республика Башкортостан',
 'Республика Бурятия',
 'Республика Дагестан',
 'Республика Ингушетия',
 'Республика Калмыкия',
 'Республика Карелия',
 'Республика Коми',
 'Республика Крым',
 'Республика Мордовия',
 'Республика Саха (Якутия)',
 'Республика Тыва',
 'Республика Хакасия',
 'Ростовская область',
 'Рязанская область',
 'Санкт-Петербург',
 'Саратовская область',
 'Сахалинская область',
 'Свердловская область',
 'Севастополь',
 'Смоленская область',
 'Ставропольский край',
 'Тамбовская область',
 'Тверская область',
 'Томская область',
 'Тульская область',
 'Тюменская область',
 'Удмуртская Республика',
 'Ульяновская область',
 'Ханты-Мансийский автономный округ - Югра',
 'Челябинская область',
 'Чеченская Республика',
 'Чувашская Республика - Чувашия',
 'Чукотский автономный округ',
 'Ямало-Ненецкий автономный округ',
 'Ярославская область']

os.makedirs('./documents/', exist_ok=True)

def search_or_save_user(mdb, effective_user):
    user = mdb.users.find_one({"userid": effective_user.id})  # поиск в коллекции users по user.id
    if not user:  # если такого нет, создаем словарь с данными
        user = {
            "userid": effective_user.id,
            "user_name": effective_user.username,
            "first_name": effective_user.first_name,
            "last_name": effective_user.last_name,
            "state": 0,
            "documents": [],
        }
        mdb.users.insert_one(user)  # сохраняем в коллекцию users
    return user

def get_state(mdb, userid):
    user = mdb.users.find_one({"userid": userid})
    if user:
        return user['state']
    return None

def set_state(mdb, userid, state):
    mdb.users.update_one({'userid': userid}, {'$set': {'state': state}})
    return True

def add_document(mdb, userid, fileid, filename):
    print(userid, fileid, filename)
    mdb.users.update_one({'userid': userid}, {'$addToSet': {'documents': {'name': filename, 'id': fileid}}})
    return True

@bot.message_handler(commands=['start'])
def start_command(message):
    user = search_or_save_user(mdb, message.from_user)
    set_state(mdb, user['userid'], 0)
    bot.send_message(message.chat.id, 'Привет. Это бот для проверки документов на коррупционные факторы. Отправь нам документ и мы вышлем экспертизу', reply_markup=startKeyboard)
    return

@bot.message_handler(func=lambda x: x.text in ['Отправить документ на проверку', 'Просмотреть отправленные документы'] and get_state(mdb, x.from_user.id) == 0)
def main_selection(message):
    user = search_or_save_user(mdb, message.from_user)
    if message.text == 'Отправить документ на проверку':
        bot.send_message(message.chat.id, 'Готов к работе. Отправьте один документ как вложение')
        set_state(mdb, user['userid'], 1)
        bot.register_next_step_handler(message, document_receiver)
    else:
        docs_inlines = types.InlineKeyboardMarkup(row_width=1)
        for i, doc in enumerate(user['documents']):
            if 'procid' in doc.keys():
                docs_inlines.add(types.InlineKeyboardButton(doc['name'], callback_data=f"senddoc|{i}"))

        bot.send_message(message.chat.id, 'Ваши обработанные документы. Нажмите на название и мы переотправим его.', reply_markup=docs_inlines)
    return

def document_receiver(message):
    user = search_or_save_user(mdb, message.from_user)
    if user['state'] != 1:
        bot.send_message(message.chat.id, 'Что то пошло не так, начните, пожалуйста, с команды /start')
    else:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = './documents/' + message.document.file_name
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        add_document(mdb, user['userid'], message.document.file_id, message.document.file_name)
        regKeyboard = types.InlineKeyboardMarkup(row_width=1)
        for reg in regions:
            regKeyboard.add(types.InlineKeyboardButton(reg, callback_data=f'reg|{reg}|{src}'))
        bot.send_message(message.chat.id, 'Выберите регион', reply_markup=regKeyboard)
    set_state(mdb, user['userid'], 0)
    return
        
def process_and_save_file(path):
    os.system(f"cp {path} .{path.split('.')[1] + '_processed.' + path.split('.')[2]}")
    return

@bot.callback_query_handler(func=lambda call: call.data.split('|')[0] == 'senddoc')
def send_document_from_inlines(call):
    user = search_or_save_user(mdb, call.from_user)
    msg = bot.send_document(call.message.chat.id, user['documents'][int(call.data.split('|')[-1])]['procid'], reply_markup=startKeyboard)
    return

@bot.callback_query_handler(func=lambda call: call.data.split('|')[0] == 'reg')
def document_processer(call):
    splt = call.split('|')
    process_document(splt[2], splt[1])
    src = splt[2]
    bot.send_message(call.message.chat.id, 'Готово!', reply_markup=startKeyboard)
    f = open('.' + src.split('.')[1] + '_processed.' + src.split('.')[2], 'rb')
    msg = bot.send_document(call.message.chat.id, f, reply_markup=startKeyboard)
    f.close()
    user = search_or_save_user(mdb, call.from_user)
    user['documents'][-1]['procid'] = msg.document.file_id
    mdb.users.update_one({'userid': user['userid']}, {'$set': {'documents': user['documents']}})
    return 

bot.polling(non_stop=True)
