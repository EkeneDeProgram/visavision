// Add an event listener that runs when the DOM content is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Fetch the unique cities from the provided text file
  axios.get("/static/data/unique_cities.txt")
    .then(response => {
      // Split the response data into an array of cities
      const cities = response.data.split("\n");
      
      // Get the city select element by its ID
      const citySelect = document.getElementById("city");
      
      // Iterate over each city and create an option element for it
      cities.forEach(city => {
        const option = document.createElement("option");
        // Set the value and text content of the option element
        option.value = city;
        option.textContent = city;
        
        // Append the option element to the city select element
        citySelect.appendChild(option);
      });
    })
    .catch(error => {
      // Log any errors that occur during the fetch
      console.error("Error fetching cities:", error);
    });

  // Fetch the unique states from the provided text file
  axios.get("/static/data/unique_states.txt")
    .then(response => {
      // Split the response data into an array of states
      const states = response.data.split("\n");
      
      // Get the state select element by its ID
      const stateSelect = document.getElementById("state");
      
      // Iterate over each state and create an option element for it
      states.forEach(state => {
        const option = document.createElement("option");
        
        // Set the value and text content of the option element
        option.value = state;
        option.textContent = state;
        
        // Append the option element to the state select element
        stateSelect.appendChild(option);
      });
    })
    .catch(error => {
      // Log any errors that occur during the fetch
      console.error("Error fetching states:", error);
    });
  
  // Get the year select element by its ID
  const yearSelect = document.getElementById("year");
  
  // Loop through the years 2024 to 2026 and create an option element for each
  for (let year = 2024; year <= 2026; year++) {
    const option = document.createElement("option");
    
    // Set the value of the option element
    option.value = year;
    
    // Set the text content of the option element
    option.textContent = year;
    
    // Append the option element to the year select element
    yearSelect.appendChild(option);
  }
});
