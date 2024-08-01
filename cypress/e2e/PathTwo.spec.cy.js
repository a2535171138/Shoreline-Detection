import 'cypress-file-upload';

Cypress.Commands.add('hideTooltips', () => {
    cy.get('body').then(($body) => {
        $body.addClass('hide-tooltip');
    });
});

beforeEach(() => {
    cy.hideTooltips();
});

describe('End-to-End Test for Entire Application', () => {
    it('Visit the app root URL', () => {
        cy.visit('http://localhost:3000');
    });

    it('Visit user guide successfully', () => {
        cy.get('button[aria-label="user guide"]').click();
        cy.contains('User Guide').click();
    });

    it('Check learn more button in user guide successfully', () => {
        cy.visit('http://localhost:3000/user-guide');

        // Click the "Learn More" link for the "View Switch" feature
        cy.contains('View Switch')
        .parents('.MuiCard-root')
        .find('button')
        .contains('Learn More')
        .click();

        // Verify that the correct dialog is opened
        cy.contains('Details'); // The title of the dialog
        cy.contains('The "View Switch" feature'); // Part of the content to verify

        // Close the dialog
        cy.get('button').contains('Close').click();
    });

    it('Return to homepage successfully', () => {
        // Visit user guide page directly in case it's not already there
        cy.visit('http://localhost:3000/user-guide');

        // Click the "BACK TO HOME PAGE" button
        cy.contains('button', 'BACK TO HOME PAGE').click();

        // Verify that we have navigated back to the home page
        cy.url().should('eq', 'http://localhost:3000/');
        cy.contains('AI-Driven Shoreline Detection');
    });
    
    it('Toggle quality check button and verify backend successfully', () => {
        // Intercept the request made to the backend when the toggle is changed
        cy.intercept('POST', '/toggle_quality_check').as('toggleQualityCheck');

        // Find the "Quality Check" switch and click it
        cy.get('div[aria-label="quality check"]').within(() => {
            cy.get('input[type="checkbox"]').click({ force: true });
        });

        // Wait for the request to be made and verify it
        cy.wait('@toggleQualityCheck').its('response.statusCode').should('eq', 200);

    });

    it('Upload images successfully', () => {
        // Open the side menu
        cy.get('button[aria-label="open drawer"]').click();

        // Click the Upload Image button
        cy.contains('Upload Image').click();

    const fileName1 = 'image1.jpg';
    const fileName2 = 'image2.jpg';
    const fileName3 = 'image3.jpg';

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

        cy.fixture(fileName3, 'binary').then(fileContent => {
            cy.get('input[type="file"]').attachFile({
            fileContent: Cypress.Blob.binaryStringToBlob(fileContent),
            fileName: fileName3,
            mimeType: 'image/jpeg'
            });
        });

        // Check if images are displayed correctly
        cy.get('img').should('have.length', 3);
    });

    it('Verify the backend response and Get result for CoastSnap successfully', () => {
        cy.get('img').should('have.length', 3);

        // Intercept the network request for the specific scene
        cy.intercept('POST', '/predict/*').as('getResult');

        // Click the Get Result button for the specific scene
        cy.contains('Get Result').click({ force: true });
        cy.contains('Choose CoastSnap Scene').click();

        // Wait for the network request to complete and get the response
        cy.wait('@getResult', { timeout: 30000 })
            .its('response.statusCode')
            .should('eq', 200);

        // Verify that the result container is displayed
        cy.get('.result-container').should('be.visible');

        // Check if the result is correctly displayed
        cy.get('.result-container', { timeout: 60000 }).should('contain.text', 'Processed Image (binary)');
    });

    it('Click Toggle All button and Switch all binary images to color images successfully', () => {
        cy.contains('Toggle All').should('be.visible');

        // Click the "Toggle All" button
        cy.contains('Toggle All').click();({ force: true });

        cy.get('.result-container').should('contain.text', 'Processed Image (color)'); 
    });

    it('Check View Log successfully', () => {
        cy.contains('View Log').should('be.visible');

        // Click the "View Log" button
        cy.contains('View Log').click();({ force: true });

        // Verify that the log view is open
        cy.get('.MuiDialog-root').should('be.visible');

        // Click the Close button
        cy.get('button[aria-label="close"]').click({ force: true });

        // Verify that the log view is closed
        cy.get('.MuiDialog-root').should('not.exist');
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