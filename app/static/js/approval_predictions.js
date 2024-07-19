// Add an event listener that runs when the DOM content is fully loaded
document.addEventListener("DOMContentLoaded", () => {
  // Get the form element by its ID
  const form = document.getElementById("predictionForm");

  // Add a submit event listener to the form
  form.addEventListener("submit", function (event) {

    // Prevent the default form submission behavior
    event.preventDefault();

    // Create an object to store form data
    const formData = {
      // Get the value of the employer input field
      employer: document.getElementById("employer").value,

      // Get the value of the city input field
      city: document.getElementById("city").value,

      // Get the value of the state input field
      state: document.getElementById("state").value,

      // Get the value of the year input field
      year: document.getElementById("year").value,

      // Get the value of the salary input field
      salary: document.getElementById("salary").value
    };

         // Send a POST request to the '/predict' endpoint with the form data
        axios.post("/predict", formData)
          .then(response => {
          // Get the element to display the prediction result
          const resultDiv = document.getElementById("predictionResult");

          // Display the approval prediction result from the response
          resultDiv.innerHTML = `Approval Prediction: ${response.data.approval_prediction}`;
        })
            
        .catch(error => {
          // Log any errors that occur during the request
          console.error("There was an error!", error);
        });
    });
});
