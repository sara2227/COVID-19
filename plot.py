import matplotlib as plt
plt.use("Agg")

def plot(model_history,epochs,model_name):
    history = model_history
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0,N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0,N), history.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")

    # save plot to disk
    plt.savefig('./figures'+model_name+'.png')
