# Updated dataset with more variation
emails = [
    {"Free": 1, "Offer": 0, "Money": 0, "Spam": 1},  # Spam
    {"Free": 0, "Offer": 1, "Money": 0, "Spam": 0},  # Not Spam
    {"Free": 0, "Offer": 0, "Money": 1, "Spam": 1},  # Spam
    {"Free": 1, "Offer": 1, "Money": 0, "Spam": 0},  # Not Spam
    {"Free": 1, "Offer": 0, "Money": 1, "Spam": 1},  # Spam
    {"Free": 0, "Offer": 1, "Money": 1, "Spam": 0},  # Not Spam
    {"Free": 1, "Offer": 1, "Money": 1, "Spam": 1},  # Spam
    {"Free": 0, "Offer": 0, "Money": 0, "Spam": 0},  # Not Spam
    {"Free": 1, "Offer": 1, "Money": 0, "Spam": 0},  # Not Spam
    {"Free": 1, "Offer": 0, "Money": 0, "Spam": 1}   # Spam
]

# New email to classify
new_email = {"Free": 0, "Offer": 1, "Money": 1}  # Free=No, Offer=Yes, Money=Yes

# Function to calculate prior probabilities P(Spam) and P(Not Spam)
def calculate_priors(data):
    total = len(data)
    spam_count = sum(email["Spam"] for email in data)
    return {
        "Spam": spam_count / total,
        "Not_Spam": (total - spam_count) / total
    }

# Function to calculate likelihoods P(Feature | Spam) and P(Feature | Not Spam)
def calculate_likelihoods(data, label):
    feature_counts = {feature: 0 for feature in data[0] if feature != "Spam"}
    total_label_count = sum(1 for email in data if email["Spam"] == label)
    
    for email in data:
        if email["Spam"] == label:
            for feature in feature_counts:
                feature_counts[feature] += email[feature]

    # Apply Laplace smoothing and calculate probabilities
    return {
        feature: (feature_counts[feature] + 1) / (total_label_count + 2) 
        for feature in feature_counts
    }

# Function to apply Naïve Bayes formula and classify an email
def classify_email(new_email, priors, likelihoods_spam, likelihoods_not_spam):
    prob_spam = priors["Spam"]
    prob_not_spam = priors["Not_Spam"]

    for feature, value in new_email.items():
        if value == 1:
            prob_spam *= likelihoods_spam[feature]
            prob_not_spam *= likelihoods_not_spam[feature]
        else:
            prob_spam *= (1 - likelihoods_spam[feature])
            prob_not_spam *= (1 - likelihoods_not_spam[feature])

    # Normalize probabilities
    total_prob = prob_spam + prob_not_spam
    return {
        "Spam": prob_spam / total_prob,
        "Not_Spam": prob_not_spam / total_prob
    }

# Compute priors and likelihoods
priors = calculate_priors(emails)
likelihoods_spam = calculate_likelihoods(emails, 1)
likelihoods_not_spam = calculate_likelihoods(emails, 0)

# Classify the new email
probabilities = classify_email(new_email, priors, likelihoods_spam, likelihoods_not_spam)

# Make a prediction
prediction = "Spam" if probabilities["Spam"] > probabilities["Not_Spam"] else "Not Spam"

# Output results
print(f"Probability of Spam: {probabilities['Spam']:.4f}")
print(f"Probability of Not Spam: {probabilities['Not_Spam']:.4f}")
print(f"Prediction: {prediction}")
