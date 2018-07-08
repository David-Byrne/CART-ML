import random
from collections import Counter


class _Question(object):
    def __init__(self, attr_id, value):
        self.attr_id = attr_id
        self.value = value
        self.numerical_mode = isinstance(self.value, float) or isinstance(self.value, int)

    def check_answer(self, instance):
        if self.numerical_mode:
            return instance[self.attr_id] >= self.value
        else:
            return instance[self.attr_id] == self.value

    def __str__(self):
        comparator = ">=" if self.numerical_mode else "=="
        return "Is attribute {} {} {}:".format(self.attr_id + 1, comparator, self.value)


class _Node(object):
    def __init__(self, question, true_subtree, false_subtree):
        self.question = question
        self.true_subtree = true_subtree
        self.false_subtree = false_subtree

    def classify(self, instance):
        if self.question.check_answer(instance):
            return self.true_subtree.classify(instance)
        else:
            return self.false_subtree.classify(instance)

    def pretty_print(self, offset=0):
        padding = "\n" + offset * "\t\t" + "└────"

        q = str(self.question)
        true_path = padding + "Y: " + self.true_subtree.pretty_print(offset=offset + 1)
        false_path = padding + "N: " + self.false_subtree.pretty_print(offset=offset + 1)

        return q + true_path + false_path


class _Leaf(object):
    def __init__(self, data):
        self.probabilities = self.class_frequency(data)

    @staticmethod
    def class_frequency(data):
        classes = [row[-1] for row in data]
        return Counter(classes)

    def classify(self, _):
        return self.probabilities

    def pretty_print(self, offset=0):
        total_count = sum(self.probabilities.values())
        results = {}
        for class_name, count in self.probabilities.items():
            results[class_name] = count * 100 / total_count
        return ",".join(["<{}:{}>".format(r[0], r[1]) for r in results.items()])


class Cart(object):

    def __init__(self, data):
        self._tree = self._create_tree(data)

    @staticmethod
    def _calc_uncertainty(data):
        # calculates Gini uncertainty for a given dataset

        uncertainty = 1
        # init uncertainty to 1 and reduce every time we hit some uncertainty
        count_of_labels = Counter([row[-1] for row in data])
        num_instances = len(data)

        for label, count in count_of_labels.items():
            prob_of_label = count/num_instances
            prob_of_matching = prob_of_label**2
            # square it as it has to be that class AND we've to pick that label
            uncertainty -= prob_of_matching

        return uncertainty

    def _calc_info_gain(self, lhs_data, rhs_data, cur_uncert):
        num_instances = len(lhs_data) + len(rhs_data)
        weight_lhs = len(lhs_data)/num_instances
        weight_rhs = len(rhs_data)/num_instances

        uncert_lhs = self._calc_uncertainty(lhs_data)
        uncert_rhs = self._calc_uncertainty(rhs_data)

        new_uncert = weight_lhs*uncert_lhs + weight_rhs*uncert_rhs
        return cur_uncert - new_uncert

    @staticmethod
    def _calc_unique_values(data, attr_id):
        values = [instance[attr_id] for instance in data]
        unique_values = set(values)
        return unique_values

    def _calc_best_split(self, data):
        best_info_gain = 0
        best_question = None

        current_uncert = self._calc_uncertainty(data)
        dimensionality = len(data[0]) - 1
        # dimensionality is 1 less than the number of entries because the last entry is the class

        for attr_id in range(dimensionality):

            unique_values = self._calc_unique_values(data, attr_id)

            for val in unique_values:

                question = _Question(attr_id, val)
                true_instances, false_instances = self._split_data(data, question)

                if max(len(true_instances), len(false_instances)) >= len(data):
                    # we didn't split the data at all so skip this one
                    continue

                info_gain = self._calc_info_gain(false_instances, true_instances, current_uncert)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_question = question

        return best_info_gain, best_question

    @staticmethod
    def _split_data(data, question):
        true_data = [row for row in data if question.check_answer(row)]
        false_data = [row for row in data if not question.check_answer(row)]
        return true_data, false_data

    def _create_tree(self, data):
        # find best way we can split the data
        info_gain, question = self._calc_best_split(data)

        if info_gain == 0:
            return _Leaf(data)

        true_data, false_data = self._split_data(data, question)

        true_subtree = self._create_tree(true_data)
        false_subtree = self._create_tree(false_data)

        return _Node(question, true_subtree, false_subtree)

    def visualise_model(self):
        print("")
        print(self._tree.pretty_print())
        print("")
        # printing blanks to get a newline on both sides of the model

    def classify(self, instance):
        result_counts = self._tree.classify(instance)
        results = []

        for class_name, count in result_counts.items():
            results += [class_name] * count
            # weighing their frequency proportionally to how confident we are it's that class

        return random.choice(results)

    def test_accuracy(self, test_data):
        num_correct = 0

        for instance in test_data:
            answer = self.classify(instance[0:-1])
            if answer == instance[-1]:
                num_correct += 1

        return num_correct/len(test_data)

    def save_actual_vs_predicted_results(self, test_data):
        with open("results.csv", mode="w") as file:
            for instance in test_data:
                answer = self.classify(instance[0:-1])
                row = ",".join([str(i) for i in instance] + [answer])
                file.write(row + "\n")
                # writes in form of attr1, attr2...attrN, actual class, predicted class
