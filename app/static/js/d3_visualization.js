// Add an event listener that runs when the DOM content is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Fetch data from the /api/analysis endpoint
  fetch("/api/analysis")
    // Convert the response to JSON
    .then(response => response.json())
    .then(data => {
      // Set the dimensions and margins for the SVG element
      const svgWidth = 1000, svgHeight = 600;
      const margin = { top: 22, right: 20, bottom: 30, left: 150 };
      const width = svgWidth - margin.left - margin.right;
      const height = svgHeight - margin.top - margin.bottom;

        // Function to create a bar chart
        const createBarChart = (data, title, xLabel, yLabel) => {
          // Append an SVG element to the #charts div
          const svg = d3.select("#charts").append("svg")
            // Set the width of the SVG
            .attr("width", svgWidth)
            // Set the height of the SVG
            .attr("height", svgHeight);
            
          // Append a group element to the SVG and transform it based on margins
          const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
            
          // Set the x and y scales
          const x = d3.scaleBand().range([0, width]).padding(0.2); 
          const y = d3.scaleLinear().range([height, 0]);
          
          // Set the domain for the x and y scales
          x.domain(Object.keys(data));
          y.domain([0, d3.max(Object.values(data))]);

          // Append the x-axis to the group element
          g.append("g")
            .attr("class", "x axis")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(x));
            
          // Append the y-axis to the group element
          g.append("g")
            .attr("class", "y axis")
            .call(d3.axisLeft(y).ticks(10));
            
          // Create the bars for the bar chart
          g.selectAll(".bar")
            .data(Object.keys(data))
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", d => x(d))
            .attr("y", d => y(data[d]))
            .attr("width", x.bandwidth())
            .attr("height", d => height - y(data[d]) - 5) 
            .attr("fill", "#1f77b4");

          // Append the chart title
          svg.append("text")
            .attr("x", (width / 2))
            .attr("y", margin.top / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(title);

          // Append the x-axis label
          svg.append("text")
            .attr("x", (width / 2))
            .attr("y", height + margin.bottom)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text(xLabel);

          // Append the y-axis label
          svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x", 0 - (height / 2))
            .attr("dy", "1em")
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text(yLabel);
        };

        // Function to create a horizontal bar chart
        const createHorizontalBarChart = (data, title, xLabel, yLabel) => {
          // Append an SVG element to the #charts div
          const svg = d3.select("#charts").append("svg")
            // Set the width of the SVG
            .attr("width", svgWidth)
            // Set the height of the SVG
            .attr("height", svgHeight);

          // Append a group element to the SVG and transform it based on margins
          const g = svg.append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

          // Set the x and y scales
          const x = d3.scaleLinear().range([0, width]);
          const y = d3.scaleBand().range([0, height]).padding(0.2);
          
          // Set the domain for the x and y scales
          x.domain([0, d3.max(Object.values(data))]);
          y.domain(Object.keys(data));

          // Append the x-axis to the group element
          g.append("g")
            .attr("class", "x axis")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(10));

          // Append the y-axis to the group element
          g.append("g")
            .attr("class", "y axis")
            .call(d3.axisLeft(y))
            .selectAll("text")
            .style("font-size", "10px")
            .call(wrap, margin.left - 10);

          // Create the bars for the horizontal bar chart
          g.selectAll(".bar")
            .data(Object.keys(data))
            .enter().append("rect")
            .attr("class", "bar")
            .attr("x", 0)
            .attr("y", d => y(d))
            .attr("width", d => x(data[d]) - 5)
            .attr("height", y.bandwidth())
            .attr("fill", "#1f77b4");

          // Append the chart title
          svg.append("text")
            .attr("x", (width / 2) + margin.left)
            .attr("y", margin.top / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(title);

          // Append the x-axis label
          svg.append("text")
            .attr("x", (width / 2) + margin.left)
            .attr("y", height + margin.top + 30)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text(xLabel);

          // Append the y-axis label
          svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x", 0 - (height / 2))
            .attr("dy", "1em")
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text(yLabel);
        };

        // Function to wrap text within a specified width     
        function wrap(text, width) {
          text.each(function () {
            var text = d3.select(this),
              words = text.text().split(/\s+/).reverse(),
              word,
              line = [],
              lineNumber = 0,
              lineHeight = 1.1, 
              y = text.attr("y"),
              dy = parseFloat(text.attr("dy")),
              tspan = text.text(null).append("tspan").attr("x", -10).attr("y", y).attr("dy", dy + "em");
            while (word = words.pop()) {
              line.push(word);
              tspan.text(line.join(" "));
              if (tspan.node().getComputedTextLength() > width) {
                line.pop();
                tspan.text(line.join(" "));
                line = [word];
                tspan = text.append("tspan").attr("x", -10).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
              }
            }
          });
        }

        // Create bar chart for Rate of Application
        createBarChart(data["Rate of Application"], "Applications by Fiscal Year", "Fiscal Year", "Number of Applications");

        // Create bar chart for Rate of Initial Approval
        createBarChart(data["Rate of Initial Approval"], "Initial Approvals by Fiscal Year", "Fiscal Year", "Number of Initial Approvals");

        // Create horizontal bar chart for Top 20 Cities by Application
        createHorizontalBarChart(data["Top 20 Cities by Application"], "Top 20 Cities by Application", "Number of Applications", "Cities");

        // Create horizontal bar chart for Top 20 Employers by Application
        createHorizontalBarChart(data["Top 20 Employers by Application"], "Top 20 Employers by Application", "Number of Applications", "Employers");

        // Create horizontal bar chart for Top 20 Cities by Approval
        createHorizontalBarChart(data["Top 20 Cities by Approval"], "Top 20 Cities by Approval", "Number of Approvals", "Cities");

        // Create horizontal bar chart for Top 20 Employers by Approval
        createHorizontalBarChart(data["Top 20 Employers by Approval"], "Top 20 Employers by Approval", "Number of Approvals", "Employers");
    });
});
