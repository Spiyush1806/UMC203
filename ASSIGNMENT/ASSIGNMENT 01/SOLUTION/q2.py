import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'oracle'))
import oracle as oracle 
import matplotlib.pyplot as plt
# from sklearn.model_selection

# Load train and test data
test_path = os.path.join(os.getcwd(), 'EMNIST', 'emnist-balanced-test.csv')
train_path = os.path.join(os.getcwd(), 'EMNIST', 'emnist-balanced-train.csv')

train, test = oracle.q2_train_test_emnist(23801, train_path, test_path)
   
y_train = train[:, 0]
X_train = train[:, 1:]
y_test = test[:, 0]
X_test = test[:, 1:]

# My numbers are 31 and 39

# Normalise pixel values
X_train = X_train / 255
X_test = X_test / 255



def naive_gaussian_bayes_confusion_matrix(X_train, X_test, y_train, y_test):
    priors = {
        31: np.mean(y_train == 31),
        39: np.mean(y_train == 39),
    }
    mu={
        31: np.mean(X_train[y_train == 31], axis=0),
        39: np.mean(X_train[y_train == 39], axis=0),
    }
    epsilon = 1e-6
    sigma = {
        31: np.var(X_train[y_train == 31], axis=0, ddof=1) + epsilon,
        39: np.var(X_train[y_train == 39], axis=0, ddof=1) + epsilon,
    }
    def eta_31(x, mu_31, sigma_31, prior_31):
        d = x.shape[0]
        log_det = np.sum(np.log(sigma_31)) + d * np.log(2 * np.pi)
        return -0.5 * log_det - 0.5 * np.sum(((x - mu_31) ** 2) / sigma_31) + np.log(prior_31)
    def eta_39(x, mu_39, sigma_39, prior_39):
        d = x.shape[0]
        log_det = np.sum(np.log(sigma_39)) + d * np.log(2 * np.pi)
        return -0.5 * log_det - 0.5 * np.sum(((x - mu_39) ** 2) / sigma_39) + np.log(prior_39)
    def predictor(X_test):
        y_pred = []
        for i in range(len(X_test)):
            eta31 = eta_31(X_test[i], mu[31], sigma[31], priors[31])
            eta39 = eta_39(X_test[i], mu[39], sigma[39], priors[39])
            eta31 = eta31 / (eta31 + eta39)
            eta39 = eta39 / (eta31 + eta39)
            diff = abs(eta31 - eta39)
            if eta31 > eta39 and diff > 2 * 0.25:
                y_pred.append(31)
            elif eta39 > eta31 and diff > 2 * 0.25:
                y_pred.append(39)
            else:
                y_pred.append(-1)
        return y_pred
    y_pred = predictor(X_test)
    true_positives = sum(1 for j in range(len(y_test)) if y_test[j] == 31 and y_pred[j] == 31)
    false_positives = sum(1 for j in range(len(y_test)) if y_test[j] == 39 and y_pred[j] == 31)
    true_negatives = sum(1 for j in range(len(y_test)) if y_test[j] == 39 and y_pred[j] == 39)
    false_negatives = sum(1 for j in range(len(y_test)) if y_test[j] == 31 and y_pred[j] == 39)
    rejects = sum(1 for j in range(len(y_test)) if y_pred[j] == -1)
    return true_positives, false_positives, true_negatives, false_negatives,rejects



def ques_3(X_train,X_test,y_train,y_test):
    # divide the train dataset into 5 folds
    groups = {}
    groups[1]=X_train[:int(len(X_train)/5)]
    groups[2]=X_train[int(len(X_train)/5):int(2*len(X_train)/5)]
    groups[3]=X_train[int(2*len(X_train)/5):int(3*len(X_train)/5)]
    groups[4]=X_train[int(3*len(X_train)/5):int(4*len(X_train)/5)]
    groups[5]=X_train[int(4*len(X_train)/5):]
    y_groups = {}
    y_groups[1]=y_train[:int(len(y_train)/5)]
    y_groups[2]=y_train[int(len(y_train)/5):int(2*len(y_train)/5)]
    y_groups[3]=y_train[int(2*len(y_train)/5):int(3*len(y_train)/5)]
    y_groups[4]=y_train[int(3*len(y_train)/5):int(4*len(y_train)/5)]
    y_groups[5]=y_train[int(4*len(y_train)/5):]
    
    confusion_matrix_storages = {}
    # find confusion matrix values for each fold
    for i in range(1,6):
        X_train = []
        y_train = []
        for j in range(1,6):
            if i!=j:
                X_train.extend(groups[j])
                y_train.extend(y_groups[j])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        tf,fp,tn,fn,rejects = naive_gaussian_bayes_confusion_matrix(X_train, X_test, y_train, y_test)
        confusion_matrix_storages[i] = (tf,fp,tn,fn)
    recalls=[]
    accuracies=[]
    precisions=[]
    f1_scores=[]
    for i in confusion_matrix_storages:
        tp,fp,tn,fn = confusion_matrix_storages[i]
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        f1_score = 2*recall*precision/(recall+precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        precisions.append(precision)
        f1_scores.append(f1_score)
        
    print('Recall',sum(recalls)/5)
    print('Accuracy',sum(accuracies)/5)
    print('Precision',sum(precisions)/5)
    print('F1 Score',sum(f1_scores)/5)
    
    # Test each of the 5 fold trained classifier on the X_test and report the number of rejects and misclassification
    # loss for each fold
    for i in range(1,6):
        X_train = []
        y_train = []
        for j in range(1,6):
            if i!=j:
                X_train.extend(groups[j])
                y_train.extend(y_groups[j])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        true_positives, false_positives, true_negatives, false_negatives,rejects = naive_gaussian_bayes_confusion_matrix(X_train, X_test, y_train, y_test)
        print('Fold:',i)
        print('Rejects:',rejects)
        print('Misclassification Loss:',(false_positives+false_negatives)/len(y_test))
        print('-------------------')

# ques_3(X_train,X_test,y_train,y_test)

# Naive Gaussian Bayes
def naive_gaussian_bayes(X_train, X_test, y_train, y_test):
    # Priors p(Y=31) and p(Y=39)
    priors = {
        31: np.mean(y_train == 31),
        39: np.mean(y_train == 39),
    }
    
    # Means mu_31 and mu_39
    mu = {
        31: np.mean(X_train[y_train == 31], axis=0),
        39: np.mean(X_train[y_train == 39], axis=0),
    }
    
    # Variance sigma_31 and sigma_39 
    epsilon = 1e-6
    sigma = {
        31: np.var(X_train[y_train == 31], axis=0, ddof=1) + epsilon,
        39: np.var(X_train[y_train == 39], axis=0, ddof=1) + epsilon,
    }

    def eta_31(x, mu_31, sigma_31, prior_31):
        d = x.shape[0]  # Number of features (should be 784 in your case)
        log_det = np.sum(np.log(sigma_31)) + d * np.log(2 * np.pi)  # Log determinant term
        return -0.5 * log_det - 0.5 * np.sum(((x - mu_31) ** 2) / sigma_31) + np.log(prior_31)

    def eta_39(x, mu_39, sigma_39, prior_39):
        d = x.shape[0]
        log_det = np.sum(np.log(sigma_39)) + d * np.log(2 * np.pi)
        return -0.5 * log_det - 0.5 * np.sum(((x - mu_39) ** 2) / sigma_39) + np.log(prior_39)

    def predictor(reject_buffer, X_test):
        y_pred = []
        for i in range(len(X_test)):
            eta31 = eta_31(X_test[i], mu[31], sigma[31], priors[31])
            eta39 = eta_39(X_test[i], mu[39], sigma[39], priors[39])
            
            # Normalise eta31 and eta39
            eta31 = eta31 / (eta31 + eta39)
            eta39 = eta39 / (eta31 + eta39)
            
            diff = abs(eta31 - eta39)
            if eta31 > eta39 and diff > 2 * reject_buffer:
                y_pred.append(31)
            elif eta39 > eta31 and diff > 2 * reject_buffer:
                y_pred.append(39)
            else:
                y_pred.append(-1)
        return y_pred
    ans=[]
    reject_buffer = [0.01, 0.1, 0.25, 0.4]
    for reject in reject_buffer:
        y_pred = predictor(reject, X_test)
        correct_preds = sum(1 for j in range(len(y_test)) if y_test[j] == y_pred[j])
        total_valid_preds = sum(1 for j in range(len(y_test)) if y_pred[j] != -1)
        rejects = sum(1 for j in range(len(y_test)) if y_pred[j] == -1)
        accuracy = correct_preds / total_valid_preds if total_valid_preds > 0 else 0
        print(f"Reject Buffer: {reject}, Accuracy: {accuracy:.4f}, Rejects: {rejects}")
        ans.append((reject, 1-accuracy,rejects))
    return ans

def sub_sampling(rat_31, rat_39, X_train, y_train):
    X_train_31 = X_train[y_train == 31]
    X_train_39 = X_train[y_train == 39]
    X_train_31 = np.random.permutation(X_train_31)
    X_train_39 = np.random.permutation(X_train_39)
    max_samples_31 = len(X_train_31)
    max_samples_39 = len(X_train_39)
    if rat_31 > rat_39:
        count_39 = min(max_samples_39, int(max_samples_31 * (rat_39 / rat_31)))
        count_31 = int(count_39 * (rat_31 / rat_39))
    else:
        count_31 = min(max_samples_31, int(max_samples_39 * (rat_31 / rat_39)))
        count_39 = int(count_31 * (rat_39 / rat_31))
    count_31 = min(count_31, max_samples_31)
    count_39 = min(count_39, max_samples_39)
    new_X_train = np.vstack((X_train_31[:count_31], X_train_39[:count_39]))
    new_y_train = np.hstack((np.full(count_31, 31), np.full(count_39, 39)))
    indices = np.random.permutation(len(new_X_train))
    new_X_train, new_y_train = new_X_train[indices], new_y_train[indices]
    print(f"Subsampling complete: 31 -> {count_31}, 39 -> {count_39}, Total -> {len(new_X_train)}")
    return new_X_train, new_y_train

def ques_2():
    splits = [[60,40], [80,20], [90,10], [99,1]]
    for split in splits:
        X_train_sub, y_train_sub = sub_sampling(split[0], split[1], X_train, y_train)
        ans = naive_gaussian_bayes(X_train_sub, X_test, y_train_sub, y_test)
        
        rejects = []
        missclassification_loss = []
        rejecteds = []
        for i in ans:
            rejects.append(i[0])
            missclassification_loss.append(i[1])
            rejecteds.append(i[2])
        
        # Plot Missclassification Loss vs Reject Buffer
        plt.figure(figsize=(10, 5))
        plt.plot(rejects, missclassification_loss)
        plt.xlabel('Reject Buffer')
        plt.ylabel('Missclassification Loss')
        plt.title(f'Missclassification Loss vs Reject Buffer (Split {split[0]}:{split[1]})')
        # for i, txt in enumerate(rejecteds):
        #     plt.annotate(f'Rejects: {txt}', (rejects[i], missclassification_loss[i]), 
        #                  textcoords="offset points", xytext=(0,10), ha='center')
        plt.show()
        
        # Plot Rejects vs Reject Buffer
        plt.figure(figsize=(10, 5))
        plt.plot(rejects, rejecteds)
        plt.xlabel('Reject Buffer')
        plt.ylabel('Rejects')
        plt.title(f'Rejects vs Reject Buffer (Split {split[0]}:{split[1]})')
        for i, txt in enumerate(rejecteds):
            plt.annotate(str(txt), (rejects[i], rejecteds[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
        plt.show()

def ques_1():
    store=naive_gaussian_bayes(X_train, X_test, y_train, y_test)
    rejects=[]
    missclassification_loss=[]
    rejecteds=[]
    for i in store:
        rejects.append(i[0])
        missclassification_loss.append(i[1])
        rejecteds.append(i[2])
    # plot the results also write the number of rejecteds at each reject buffer
    plt.plot(rejects,missclassification_loss)
    plt.xlabel('Reject Buffer')
    
    plt.ylabel('Missclassification Loss')
    plt.title('Missclassification Loss vs Reject Buffer')
    plt.show()
    plt.plot(rejects,rejecteds)
    plt.xlabel('Reject Buffer')
    plt.ylabel('Rejects')
    plt.title('Rejects vs Reject Buffer')
    plt.show()

# ques_1()
ques_2()
# ques_3(X_train,X_test,y_train,y_test)
# ques_4(X_train,X_test,y_train,y_test)