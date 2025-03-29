# Team 21
## Preprocessing and Classification
### Work Split:
Raghuveer: Preprocessing and Logistic Regression

Vijwal: KNN (+Mahalanobis variant)

Aariz: SVM

Abhinandan: Neural Network

### CSV Files:
|Name|Description|
|---|---|
|21.csv|Given file|
|Str_To_Int.csv|Converted string classes to integers|
|One-Hot.csv|Converted string classes to different one-hot columns|
|_Orig_Int|Original data with string-integer replacements|
|_Scaled_Int|Standardized data with string-integer replacements|
|_Orig_OH|Original data with one-hot encoding|
|_Scaled_OH|Standardized data with one-hot encoding|

#### String replacements:
| Int | EmploymentStatus |
| --- | --- |
|0|Employed|
|1|Self-Employed|
|2|Unemployed|

| Int | EducationLevel |
| --- | --- |
|0|Master|
|1|Associate|
|2|Bachelor|
|3|High School|
|4|Doctorate|

| Int | MaritalStatus |
| --- | --- |
|0|Married|
|1|Single|
|2|Divorced|
|3|Widowed|

| Int | HomeOwnershipStatus |
| --- | --- |
|0|Own|
|1|Mortgage|
|2|Rent|
|3|Other|

| Int | LoanPurpose |
| --- | --- |
|0|Home|
|1|Debt Consolidation|
|2|Education|
|3|Other|
|4|Auto|
