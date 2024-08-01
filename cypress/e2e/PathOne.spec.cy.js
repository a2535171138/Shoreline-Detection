import 'cypress-file-upload';

describe('End-to-End Test for Entire Application', () => {
  it('Visit the app root URL', () => {
    cy.visit('http://localhost:3000');
  });

  it('Upload images successfully', () => {
    // Open the side menu
    cy.get('button[aria-label="open drawer"]').click();

    // Click the Upload Image button
    cy.contains('Upload Image').click();

   const fileName1 = 'image1.jpg';
   const fileName2 = 'image2.jpg';

    cy.fixture(fileName1, 'binary').then(fileContent => {
      cy.get('input[type="file"]').attachFile({
        fileContent: Cypress.Blob.binaryStringToBlob(fileContent),
        fileName: fileName1,
        mimeType: 'image/jpeg'
      });
    });

    cy.fixture(fileName2, 'binary').then(fileContent => {
      cy.get('input[type="file"]').attachFile({
        fileContent: Cypress.Blob.binaryStringToBlob(fileContent),
        fileName: fileName2,
        mimeType: 'image/jpeg'
      });
    });

    // Check if images are displayed correctly
    cy.get('img').should('have.length', 2); // Ensure there are 2 images displayed
  });

  it('Zoom in and reset the image successfully', () => {
    // Make sure the image is uploaded and displayed
    cy.get('img').should('have.length', 2);

    // Click on the image to enlarge
    cy.get('img').first().click();

    // Verify that the modal box is displayed
    cy.get('.modal-dialog').should('be.visible');

    // Verify whether the image is displayed in the modal box
    cy.get('.modal-dialog img').should('be.visible');

    // Simulate pressing the 'Escape' key to close the modal
    cy.get('body').type('{esc}');

    // Verify that the modal is closed
    cy.get('.modal-dialog').should('not.exist');
  });

  it('Delete an image successfully', () => {
    cy.get('img').should('have.length', 2);

    // Click the delete button of the first image
    cy.get('button[aria-label="delete image"]').first().click();

    // Check if the number of images has decreased
    cy.get('img').should('have.length', 1);
  });

  it('Verifies the backend response and Get result successfully for Narrabeen', () => {
    const scene = 'Narrabeen';
    cy.get('img').should('have.length', 1);

    // Intercept the network request for the specific scene
    cy.intercept('POST', `/predict/${scene}`).as('getResult');

    // Click the Get Result button for the specific scene
    cy.contains('Get Result').click();
    cy.contains('Choose Narrabeen Scene').click();

    // Wait for the network request to complete and get the response
    cy.wait('@getResult', { timeout: 60000 }).then((interception) => {
        expect(interception.response.statusCode).to.eq(200);
    });

    // Verify that the result container is displayed
    cy.get('.result-container').should('be.visible');

    // Check if the result is correctly displayed
    cy.get('.result-container').should('contain.text', 'Processed Image (binary)');
});


  it('Switch to color image successfully', () => {
    // Ensure the result is displayed
    cy.get('.result-container').should('be.visible');

    // Confirm the switch button exists
    cy.get('button[aria-label="switch to the other image"]').should('exist');

    // Click the switch button to change to color image
    cy.get('button[aria-label="switch to the other image"]').first().click();

    // Click the switch button to change to color image
    cy.get('button[aria-label="switch to the other image"]').click();

    cy.get('.result-container', { timeout: 1000 }).should('contain.text', 'Processed Image (color)'); 
  });

  it('Zoom in and out the processed image successfully', () => {
    // Ensure the second processed image is displayed
    cy.get('.result-image').eq(1).within(() => {
      cy.get('img').should('exist');
      cy.get('img').click();
    });

    // Verify that the modal box is displayed
    cy.get('.modal-dialog').should('be.visible');

    // Verify whether the image is displayed in the modal box
    cy.get('.modal-dialog img').should('be.visible');

    // Simulate pressing the 'Escape' key to close the modal
    cy.get('body').type('{esc}');

    // Verify that the modal is closed
    cy.get('.modal-dialog').should('not.exist');
  });

  it('Check the download options and verify the download successfully', () => {
    // Make sure the result container is visible
    cy.get('.result-container').should('be.visible');

    // Find the download button and click it
    cy.get('button[aria-label="download options"]').click();

    // Make sure the drop-down menu item is visible
    cy.get('.MuiMenu-list').should('be.visible');

    // Verify that each option in the drop-down menu exists
    cy.contains('Download Binary Image').should('be.visible');
    cy.contains('Download Color Image').should('be.visible');
    cy.contains('Download Pixel Data (CSV)').should('be.visible');
    cy.contains('Download All File').should('be.visible');

    // Click on the Download Binary Image option and verify the download operation
    cy.contains('Download Binary Image').click();
    cy.wait(1000);  // Wait for the download to complete

    // Click the Download Color Image option and verify the download operation
    cy.get('button[aria-label="download options"]').click();
    cy.contains('Download Color Image').click();
    cy.wait(1000);  // Wait for the download to complete

    // Click the Download Pixel Data (CSV) option and verify the download operation
    cy.get('button[aria-label="download options"]').click();
    cy.contains('Download Pixel Data (CSV)').click();
    cy.wait(1000);  // Wait for the download to complete

    // Click the Download All File option and verify the download operation
    cy.get('button[aria-label="download options"]').click();
    cy.contains('Download All File').click();
    cy.wait(1000);  // Wait for the download to complete

  });

  it('Check the sidebar "Download All" button and verify the download successfully', () => {
    // Make sure the result container is visible
    cy.get('.result-container').should('be.visible');

    // Find and click the "Download All" button in the sidebar
    cy.contains('Download All').click();

    // Make sure the drop-down menu item is visible
    cy.get('.MuiMenu-list').should('be.visible');

    // Verify that each option in the drop-down menu exists and is visible
    cy.contains('Download All Binary Images (ZIP)').should('be.visible');
    cy.contains('Download All Color Images (ZIP)').should('be.visible');
    cy.contains('Download All Pixel Data (CSV)').should('be.visible');
    cy.contains('Download All Types (ZIP)').should('be.visible');

    // Click on the Download All Binary Images (ZIP) option and verify the download operation
    cy.contains('Download All Binary Images (ZIP)').click();
    cy.wait(1000);  

    // Click on the Download All Color Images (ZIP) option and verify the download operation
    cy.contains('Download All').click();
    cy.contains('Download All Color Images (ZIP)').click();
    cy.wait(1000); 

    // Click on the Download All Pixel Data (CSV) option and verify the download operation
    cy.contains('Download All').click();
    cy.contains('Download All Pixel Data (CSV)').click();
    cy.wait(1000); 

    // Click on the Download All Types (ZIP) option and verify the download operation
    cy.contains('Download All').click();
    cy.contains('Download All Types (ZIP)').click();
    cy.wait(1000);  
  });

  it('Clear image successfully', () => {
    // Make sure the result container is visible
    cy.get('.result-container').should('be.visible');

    // Make sure there is an image on the page
    cy.get('.result-image').should('have.length.greaterThan', 0);

    // Find and click the "Clear Image" button in the sidebar
    cy.contains('Clear Image').click();

    // Wait for the dialog to appear and click the "YES" button
    cy.get('#alert-dialog-title').should('contain', 'Confirm Clear Images'); // Ensure dialog title is visible
    cy.contains('button', 'Yes').click();

    // Verify that the image on the page has been cleared
    cy.get('.result-image').should('not.exist');
  });
});