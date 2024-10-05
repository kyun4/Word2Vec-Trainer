import sys
from gensim.models import Word2Vec
import csv

job_data = [['python','developer','backend','api','rest'],
            ['javascript','frontend','react','nodejs'],
            ['solidworks','mechanical','engineer','flowsimulation','sketch','3dmodelling'],
            ['graphic','artist','photoshop','illustrator','canva'],
            ['project','manager','agile','devops','scrum','leadership']]

with open('skill2vec_10K.csv', mode = 'r') as file:
    csv_reader = csv.reader(file)

    next(csv_reader)

    for row in csv_reader:
        array_row = list(row)
        array_content = []
        for row_item in array_row:
            
            if row_item != '' or row_item != None:
                if(row_item != ''):
                    array_content.append(row_item)
                   
        job_data.append(array_content)
        

model = Word2Vec(sentences=job_data, vector_size=100,window=5,min_count=1, workers=4)

model.save("job_skills_word2vec.model")

