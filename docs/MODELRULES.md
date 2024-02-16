# Model Creation Guidelines

Thank you for your interest in contributing to our repository! Below are the guidelines for creating models:

1. **Flexibility**: You are encouraged to design and implement models as you see fit. There are no restrictions on the architecture or techniques used.

2. **Self-contained Class**: Each model should be encapsulated within a self-contained class. This class should include, at minimum, the following methods:
   - `train_model`: This method should contain the necessary code for training the model. It should accept training data as input and update the model parameters accordingly.
   - `sample` (optional): If applicable, this method should generate samples from the trained model. It may take additional parameters for controlling the sampling process.

3. **Documentation**: Provide clear and concise documentation within the class to explain its functionality and usage. Document any additional features or functionalities beyond the basic training and sampling methods.

4. **Additional Features**: You are welcome to implement additional features within the model class, such as outlier detection, data augmentation, or specialized sampling techniques. Document these features thoroughly to facilitate understanding and usage.

5. **Readability and Maintainability**: Write clean, readable, and well-commented code to ensure that others can easily understand and modify your implementation if needed.

6. **Dependencies**: Minimize external dependencies and ensure that all required libraries are listed in the repository's `requirements.txt` file.

7. **Testing**: Whenever possible, include unit tests to verify the correctness and robustness of your implementation.

8. **Licensing**: Ensure that your code complies with the repository's licensing terms and that you have the necessary permissions to contribute it.

By following these guidelines, you can create models that are easy to understand, use, and integrate into our repository. Thank you for your contributions!
