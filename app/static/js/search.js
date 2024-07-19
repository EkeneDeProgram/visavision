// Add an event listener to the search input that triggers on keyup
document.getElementById("searchInput").addEventListener("keyup", function () {
  // Get the current value of the search input and convert it to lowercase
  const searchTerm = this.value.toLowerCase();
  
  // Get all the list items within the employers list
  const employers = document.querySelectorAll("#employersList li");

  // Iterate over each employer in the list
  employers.forEach(function(employer) {
    // Get the text content of the current employer and convert it to lowercase
    const text = employer.textContent.toLowerCase();
    
    // Check if the employer text includes the search term
    if (text.includes(searchTerm)) {
      // If it does, make sure the employer is displayed  
      employer.style.display = "";
    } else {
      // If it doesn't, hide the employer
      employer.style.display = "none";
    }
  });
});

