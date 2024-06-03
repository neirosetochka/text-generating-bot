import telebot
from telebot import types
from model_wrapper import ModelWrapper
import re

bot = telebot.TeleBot('6815921534:AAGKT_yQydCGTToUn9DTOidxDoV40blPbxE')
model_wrapper = ModelWrapper()
past_text = None

@bot.message_handler(commands=['start'])
def start(message):     
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True) #создание новых кнопок
    btn = types.KeyboardButton('Разбудить меня🐧')
    markup.add(btn)
    bot.send_message(message.from_user.id, "Привет! Я Лиса (ударение, кстати, на «и»), LM-модель на основе n-gram для генерации текста ✌😇😺. Я пока еще сплю 😴", reply_markup=markup)


@bot.message_handler(commands=['help'])
def help(message):
    help_message = """Мои команды:
/start - запуск бота.
/help - список всех команд
/params - посмотреть текущие параметры генерации
/repeat - повторить генерацию предложения (чтобы искать подходящие параметры)
temperature = value - установит температуру = value
"""
    bot.send_message(message.from_user.id, help_message)

def bot_generate(message, text):

    if not model_wrapper.model.tokenizer.text_preprocess(text):
            bot.send_message(message.from_user.id, 'Нужен хотя бы один русский символ💔')

    else:
        status, result = model_wrapper.generate(text)
        if status:
            bot.send_message(message.from_user.id, result)
        else:
            bot.send_message(message.from_user.id, 'Ошибка генерации😱')


@bot.message_handler(commands=['repeat'])
def repeat(message):
    if past_text is None:
        bot.send_message(message.from_user.id, 'Мне пока нечего повторять.')
    else:
        bot_generate(message, past_text)

@bot.message_handler(commands=['params'])
def params(message):
    config = model_wrapper.generate_kwargs['generation_config']
    temperature = config.temperature
    sample_top_p = config.sample_top_p
    bot.send_message(message.from_user.id, 'temperature = ' + str(temperature) + ', sample_top_p =' + str(sample_top_p) + '.')
        
@bot.message_handler(content_types=['text'])
def get_text_messages(message):

    
    if message.text == 'Разбудить меня🐧':

        status = model_wrapper.load('StatLM')
        if status:
            bot_message = "Я проснулась 💃🎶💃"
            s1 = ".\nЕсли хочешь поменять параметры генерации (temperature/sample_top_p), напиши так: «название_параметра = значение»🍒.\nЕсли хочешь узнать текущее значение параметров, напиши /params."
            s2 = '\nЕсли хочешь повторить прошлую генерацию, напиши /repeat.'
            bot_message += s1 + s2
            bot.send_message(message.from_user.id, bot_message)
        else:
            bot.send_message(message.from_user.id, "Что-то не хочу просыпаться, такие сны приятные... 😭😭😭")   
    
    elif 'temperature' in message.text or 'sample_top_p' in message.text:
        value = message.text.split()[2]
        success = False
        if re.match(r'\d+.\d+', value):
            value = float(value)
            success = True
        elif re.match(r'\d+,\d+', value):
            value = float(value.replace(',', '.'))
            success = True
        elif re.match(r'\d+(.\d+){0,1}e-{0,1}\d+', value):
            v0, v1 = [float(x) for x in value.split('e')]
            value = v0 * 10 ** v1
            success = True
        
        if success:
            result = model_wrapper.change_kwargs(value, message.text.split()[0])
            if result:
                bot.send_message(message.from_user.id, "Значение изменено😇")
            else:
                bot.send_message(message.from_user.id, "Не знаю такой параметр😥")
        else:
            bot.send_message(message.from_user.id, "Что-то пошло не так😔")

    else:
        past_text = message.text + '\n##'
        bot_generate(message, message.text)
        
bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть
