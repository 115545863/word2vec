import os

with open('human_chat.txt',encoding='utf-8') as file:
    contents = file.read()
    contents = contents.replace('Human 1: ','')
    contents = contents.replace('Human 2: ','')
    contents = contents.rstrip()
    with open('human_chat_after.txt','w',encoding='utf-8') as new_file:
        new_file.write(contents)
        new_file.close()
    file.close()