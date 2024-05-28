import telebot
from telebot import types
from model_wrapper import ModelWrapper
import re

bot = telebot.TeleBot(your_token)
model_wrapper = ModelWrapper()
change_str = None

@bot.message_handler(commands=['start'])
def start(message):     
    bot.send_message(message.from_user.id, "Привет! Я Лиса (ударение, кстати, на «и»), созданная специально для курса «Современный NLP. Большие языковые модели» от VK Education ✌😇😺. Напиши /help, чтобы познакомиться с моими возможностями💕.")

@bot.message_handler(commands=['help'])
def help(message):
    help_message = """Мои команды:
/start старт бота
/model выбор модели
/checkmodel посмотреть, как модель сейчас загружена
/generate сгенерировать текст по контексту (можно использовать без введения команды)
"""
    bot.send_message(message.from_user.id, help_message)

@bot.message_handler(commands=['model'])
def model(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1) #создание новых кнопок
    btn1 = types.KeyboardButton('StatLM')
    btn2 = types.KeyboardButton('GPT')
    btn3 = types.KeyboardButton('Llama')
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "Выбери модель генерации текста 👀", reply_markup=markup)
        
@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    
    if message.text in ['StatLM', 'GPT', 'Llama']:

        status, result = model_wrapper.load(message.text)
        if status:
            bot.send_message(message.from_user.id, "Подгружено")
        else:
            bot.send_message(message.from_user.id, "Что-то не хочет подгружаться :(")    
    
    elif message.text in ['Поменять temperature', 'Поменять sample_top_p']:

        bot.send_message(message.from_user.id, "Жду новое значение :)")
        change_str = message.text.split()[1]
    
    elif re.match(r'^-?\d+(?:\.\d+)$', message.text):
        value = float(message.text)
        model_wrapper.change_kwargs(value, change_str)

    else:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1) #создание новых кнопок
        btn1 = types.KeyboardButton('Поменять temperature')
        btn2 = types.KeyboardButton('Поменять sample_top_p')
        markup.add(btn1, btn2)
        status, result = model_wrapper.generate(message.text)

        if status:
            bot.send_message(message.from_user.id, result, reply_markup=markup)
        
bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть
