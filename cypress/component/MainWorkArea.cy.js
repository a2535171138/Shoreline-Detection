import React from 'react';
import MainWorkArea from '../../src/components/MainWorkArea';

describe('MainWorkArea Component', () => {
  const uploadedImageFiles = [
    {
      file: {
        name: 'image1.jpg',
        lastModified: new Date().getTime(),
      },
      url: 'cypress/fixtures/image1.jpg',
    },
    {
      file: {
        name: 'image2.jpg',
        lastModified: new Date().getTime(),
      },
      url: 'cypress/fixtures/image2.jpg',
    },
  ];

  const predictionResults = [
    {
      binaryResult: 'https://via.placeholder.com/150/0000FF',
      colorResult: 'https://via.placeholder.com/150/FF0000',
      pixelsResult: 'some-pixels-data',
      processingTime: new Date().toISOString(),
    },
    'error',
  ];

  const showResults = true;

  let onDeleteImage;
  let displayModes;
  let setDisplayModes;

  beforeEach(() => {
    onDeleteImage = cy.stub();
    displayModes = ['binary', 'color'];
    setDisplayModes = cy.stub();

    cy.fixture('image1.jpg').then((image1) => {
      cy.fixture('image2.jpg').then((image2) => {
        uploadedImageFiles[0].url = URL.createObjectURL(new Blob([image1]));
        uploadedImageFiles[1].url = URL.createObjectURL(new Blob([image2]));

        cy.mount(
          <MainWorkArea
            uploadedImageFiles={uploadedImageFiles}
            predictionResults={predictionResults}
            showResults={showResults}
            onDeleteImage={onDeleteImage}
            displayModes={displayModes}
            setDisplayModes={setDisplayModes}
          />
        );
      });
    });
  });

  it('renders images correctly', () => {
    cy.get('img').should('have.length', 4);
  });

  it('opens and closes the modal', () => {
    cy.get('img').first().click();
    cy.get('.modal').should('be.visible');
    cy.get('body').type('{esc}');
    cy.get('.modal').should('not.exist');
  });
});

