import random
from statistics import mean, stdev
from cart import Cart


def main():
    # read in and preprocess data
    with open("owls15.csv") as file:
        content = file.readlines()

    data = []

    for entry in content:
        readings = entry.rstrip("\n").split(",")
        attributes = [float(r) for r in readings[0:-1]]
        attributes.append(readings[-1])
        # add back in the classes
        data.append(attributes)

    # build and print CART decision tree
    Cart(data).visualise_model()

    # test accuracy 10 times and display the results
    acc_scores = []
    for i in range(0,10):
        random.shuffle(data)
        split = len(data) * 2//3
        training_data = data[0:split]
        test_data = data[split:]
        cart = Cart(training_data)
        accuracy = cart.test_accuracy(test_data)
        print("Round {}, Accuracy is {:.4f}".format(i+1, accuracy))
        acc_scores.append(accuracy)
        cart.save_actual_vs_predicted_results(test_data)

    print("")
    print("CART classifier has an accuracy of {:.2f}%, +/- {:.2f}%"
          .format(mean(acc_scores) * 100, stdev(acc_scores) * 2 * 100))
    # Giving +/- 2 standard deviations to show confidence interval of result

if __name__ == '__main__':
    main()
