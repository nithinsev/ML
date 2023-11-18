def find_s_algorithm(training_data):
    hypothesis = training_data[0][:-1] 

    for example in training_data:
        if example[-1] == 'Y':  
            for i in range(len(hypothesis)):
                if example[i] != hypothesis[i]:
                    hypothesis[i] = '?'     
    return hypothesis
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Y'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Y'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'N'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Y']
]
result_hypothesis = find_s_algorithm(training_data)
print("Final Hypothesis:", result_hypothesis)
