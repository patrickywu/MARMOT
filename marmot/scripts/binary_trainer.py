from marmot.data.text_processor import text_processor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup
import time
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score

# a modification of run_glue.py, found here: https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py
def binary_trainer(model, train_dataset, validation_dataset, epochs, learning_rate, batch_size,
                  gradient_clipping=False, proportion_warmup_steps=0.1, weight=None, device='cuda'):

    optimizer = AdamW(model.parameters(),
                      lr = learning_rate,
                      betas = (0.9,0.98),
                      eps = 1e-8)

    train_dataloader = DataLoader(train_dataset,
                                  sampler = RandomSampler(train_dataset),
                                  batch_size = batch_size)

    validation_dataloader = DataLoader(validation_dataset)

    total_steps = epochs*len(train_dataloader)

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=proportion_warmup_steps*total_steps, num_training_steps=total_steps, num_cycles=0.5)

    tp = text_processor(device=device)

    # Training
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        train_loss = 0

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and step != 0:
                print("Batch {} of {}. Elapsed: {:.4f} seconds".format(step, len(train_dataloader), time.time() - start_time))

            model.zero_grad()

            caption_ii, caption_tti, caption_am = tp.extract_bert_inputs(batch['image_caption'])
            text_ii, text_tti, text_am = tp.extract_bert_inputs(batch['text'])

            img = batch['image'].to(device)
            label = batch['label'].to(device)

            out, _ = model(img=img, caption_ii=caption_ii, caption_tti=caption_tti, caption_am=caption_am, text_ii=text_ii, text_tti=text_tti, text_am=text_am)
            loss = F.cross_entropy(input=out.view(-1,2), target=label, weight=weight, reduction='mean')

            train_loss += loss.item()

            loss.backward()
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradients issue
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_training_loss = train_loss / len(train_dataloader)
        print('(Epoch {} / {}): loss: {:.4f}'.format(epoch+1, epochs, avg_training_loss))
        print('Time for Epoch: {:.4f}'.format(time.time() - start_time))

        # Validation
        model.eval()

        val_loss = 0

        y_predicted = []
        y_proba = []
        y_true = []

        for val in validation_dataloader:
            caption_ii, caption_tti, caption_am = tp.extract_bert_inputs(val['image_caption'])
            text_ii, text_tti, text_am = tp.extract_bert_inputs(val['text'])

            img = val['image'].to(device)
            val_label = val['label'].to(device)

            with torch.no_grad():
                val_out, _ = model(img, caption_ii, caption_tti, caption_am, text_ii, text_tti, text_am)
                val_out = val_out.view(-1,2)
                val_proba = F.softmax(val_out, dim=1)
                val_predicted = torch.argmax(val_out, dim=1)
                val_loss_ind = F.cross_entropy(input=val_out, target=val_label, weight=weight, reduction='mean').item()
            # Update y_predicted and y_true
            y_predicted.append(val_predicted)
            y_proba.append(val_proba[:,1])
            y_true.append(val_label)

            # Validation loss
            val_loss += val_loss_ind

        y_predicted = torch.cat(y_predicted)
        y_proba = torch.cat(y_proba)
        y_true = torch.cat(y_true)

        # Metrics
        val_accuracy = accuracy_score(y_true.to('cpu'), y_predicted.to('cpu'))
        print("Validation Accuracy: {:.4f}".format(val_accuracy))

        val_f1_weighted = f1_score(y_true.to('cpu'), y_predicted.to('cpu'), average='weighted')
        print("Validation F1 Score (Weighted): {:.4f}".format(val_f1_weighted))

        val_f1_macro = f1_score(y_true.to('cpu'), y_predicted.to('cpu'), average='macro')
        print("Validation F1 Score (Macro): {:.4f}".format(val_f1_macro))

        val_f1 = f1_score(y_true.to('cpu'), y_predicted.to('cpu'), average=None)
        print("Validation F1 Scores (By Class): Not Hit: {:.4f}, Hit: {:.4f}".format(*val_f1))

        val_precision = precision_score(y_true.to('cpu'), y_predicted.to('cpu'), average=None)
        print("Precision (By Class): Not Hit: {:.4f}, Hit: {:.4f}".format(*val_precision))

        val_recall = recall_score(y_true.to('cpu'), y_predicted.to('cpu'), average=None)
        print("Recall (By Class): Not Hit: {:.4f}, Hit: {:.4f}".format(*val_recall))

        val_rocauc = roc_auc_score(y_true.to('cpu'), y_proba.to('cpu'))
        print("ROC AUC: {:.4f}".format(val_rocauc))

        avg_val_loss = val_loss / len(validation_dataloader)
        print("Validation Loss: {:.4f}".format(avg_val_loss))

    return val_accuracy, val_f1_weighted, val_f1_macro, val_f1, val_precision, val_recall, val_rocauc, avg_val_loss
