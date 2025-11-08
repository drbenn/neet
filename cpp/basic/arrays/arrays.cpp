#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// 1. Define the Class
class ArrayProblems {
  public:
  // 2. Define a function (method) within the class that returns a string
  std::string generateLogMessage() {
    return "Hello C++ World!";
  }

  bool containsDuplicateSort(std::vector<int> input) {
    // Sort the input vector in place (C++ equivalent of input.sort())
    std::sort(input.begin(), input.end());

    int recent = 0;
    bool is_first = true;

    // newer/safer js/cpp for...of loop
    // for (int current_num : input) { // Range-based for loop (modern C++ equivalent of JS for...of)
    //   if (!is_first && recent == current_num) {
    //     // duplicate found
    //     return true;
    //   }

    //   recent = current_num;
    //   is_first = false;
    // }

    // older/more control for loop
    /**
     * NOTE:: Conclusion: Array elements start at index 0, but the loop counter starts at 1 
     * to perform a valid comparison of the second element (input[1]) with the first element 
     * (input[0]), thereby skipping the unnecessary and dangerous step of trying to process the 
     * first element with no preceding value. It's a fundamental, efficient C++ pattern, 
     * not "noob bullshit."
     */
    for (size_t i = 1; i < input.size(); i++) {
      if (input[i] == input[i-1]) {
        return true;
      }
    }

    // no duplicates found
    return false;
  }
};


int main () {
  // Create an instance (object) of the ArrayProblems class
  ArrayProblems probs;

  // Call the function (method) on the object and store the returned string
  std::string message = probs.generateLogMessage();

  std::vector<int> test_data = {5,2,8,1,9,9};
  bool sortReturn = probs.containsDuplicateSort(test_data);

  // Log the returned message to the console
  std::cout << sortReturn << std::endl;

  // Return 0 to indicate successful execution
  return 0;
}