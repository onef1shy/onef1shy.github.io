/* ==========================================================================
   Publication Toggle Script
   Toggle between showing representative publications and all publications
   ========================================================================== */

$(document).ready(function(){
  // Get all paper boxes
  var $paperBoxes = $('.paper-box[data-representative]');
  
  // Get toggle links
  var $toggleRepresentative = $('#publication-toggle');
  var $toggleAll = $('#publication-toggle-all');
  
  // Check if we have paper boxes
  if ($paperBoxes.length === 0) {
    // Hide toggle links if no publications
    $toggleRepresentative.hide();
    $toggleAll.hide();
    return;
  }
  
  // Load saved preference or default to 'all'
  var savedMode = localStorage.getItem('publicationMode');
  var currentMode = (savedMode === 'all' || savedMode === 'representative') ? savedMode : 'all';
  
  // Initialize display
  initializeDisplay();
  updateLinkStates();
  
  // Handle toggle link clicks
  $toggleRepresentative.on('click', function(e) {
    e.preventDefault();
    if (currentMode !== 'representative') {
      currentMode = 'representative';
      updateDisplay();
      updateLinkStates();
      localStorage.setItem('publicationMode', 'representative');
    }
  });
  
  $toggleAll.on('click', function(e) {
    e.preventDefault();
    if (currentMode !== 'all') {
      currentMode = 'all';
      updateDisplay();
      updateLinkStates();
      localStorage.setItem('publicationMode', 'all');
    }
  });
  
  // Initialize display on page load (without animation)
  function initializeDisplay() {
    $paperBoxes.each(function() {
      var $paperBox = $(this);
      var isRepresentative = $paperBox.attr('data-representative') === 'true';
      
      if (currentMode === 'representative') {
        // Show only representative papers
        if (isRepresentative) {
          $paperBox.show();
        } else {
          $paperBox.hide();
        }
      } else {
        // Show all papers
        $paperBox.show();
      }
    });
  }
  
  // Update display based on current mode (with animation)
  function updateDisplay() {
    $paperBoxes.each(function() {
      var $paperBox = $(this);
      var isRepresentative = $paperBox.attr('data-representative') === 'true';
      var isVisible = $paperBox.is(':visible');
      
      if (currentMode === 'representative') {
        // Show only representative papers
        if (isRepresentative && !isVisible) {
          $paperBox.slideDown(300);
        } else if (!isRepresentative && isVisible) {
          $paperBox.slideUp(300);
        }
      } else {
        // Show all papers
        if (!isVisible) {
          $paperBox.slideDown(300);
        }
      }
    });
  }
  
  // Update link states
  function updateLinkStates() {
    if (currentMode === 'representative') {
      $toggleRepresentative.addClass('active');
      $toggleAll.removeClass('active');
    } else {
      $toggleRepresentative.removeClass('active');
      $toggleAll.addClass('active');
    }
  }
});

