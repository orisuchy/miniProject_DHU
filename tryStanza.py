import stanza

stanza.download('he')
nlp = stanza.Pipeline('he', processors='tokenize, pos')
doc = nlp("אני כל כך עצוב לי וגשם על העיר ודיזינגוף נראה לי כמו רכבת לילה לקהיר")
doc.sentences[0].print_dependencies()
print(doc.sentences[0])