import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from DoubleLayer import DoubleLayer  # Make sure you have your model class correctly imported

# Load your dataset
print("Loading dataset...")
data = pd.read_csv("Train_Scaled_Int.csv")  # Ensure your file path is correct

# Check the columns in the dataset
print("Columns in the dataset:")
print(data.columns)

# Strip leading/trailing spaces from column names (if any)
data.columns = data.columns.str.strip()

# Feature and target columns
feature_names = [
    'Age', 'AnnualIncome', 'CreditScore', 'EmploymentStatus', 'EducationLevel',
    'Experience', 'LoanAmount', 'LoanDuration', 'MaritalStatus',
    'NumberOfDependents', 'HomeOwnershipStatus', 'MonthlyDebtPayments', 
    'CreditCardUtilizationRate', 'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
    'DebtToIncomeRatio', 'BankruptcyHistory', 'LoanPurpose', 'PreviousLoanDefaults', 
    'PaymentHistory', 'LengthOfCreditHistory', 'SavingsAccountBalance', 
    'CheckingAccountBalance', 'TotalAssets', 'TotalLiabilities', 'MonthlyIncome', 
    'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth', 'BaseInterestRate', 
    'InterestRate', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio'
]

# Separate features and target
X_new = data.drop('LoanApproved', axis=1)
X = X_new.drop('RiskScore', axis=1).values
y = data['LoanApproved'].values  # Adjust if needed

# Check the shapes of the input data
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Ensure the correct shape
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Check the shapes of the tensors
print(f"X_train_tensor shape: {X_train_tensor.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")

# Initialize the model
input_size = X_train.shape[1]  # Number of features
model = DoubleLayer(input_size=input_size, hidden1=64, hidden2=64)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification, use BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
print("Starting training...")

try:
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)

        # Compute loss
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

except Exception as e:
    print(f"Error during training: {e}")

# Save the trained model
torch.save(model.state_dict(), "best_model.pt")
print("Model saved as 'best_model.pt'")
