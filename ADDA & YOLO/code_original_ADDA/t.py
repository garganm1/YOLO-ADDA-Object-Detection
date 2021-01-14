with open('./checkpoints/confthres_0.9_lr_0.01/source_encoder/validation/val_recalls.txt', 'r') as f:
    scores = f.read().split('\n')[:-1]
    recall = [float(score) for score in scores]
    
with open('./checkpoints/confthres_0.9_lr_0.01/source_encoder/validation/val_F1.txt', 'r') as g:
    f1 = g.read().split('\n')[:-1]
    f1 = [float(s) for s in f1]

max_f1 = 0 
epoch = 0 
for i in range(len(recall)):
    if recall[i] > 0.85:
        if f1[i] > max_f1:
            max_f1 = f1[i]
            epoch = i

print(epoch)




