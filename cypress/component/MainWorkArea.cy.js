/// <reference types="cypress" />
import React from 'react';
import { mount } from '@cypress/react';
import MainWorkArea from '../../src/components/MainWorkArea';

describe('MainWorkArea Component', () => {

  it('displays uploaded images', () => {
    const mockUploadedFiles = [
      {
        file: new File([''], 'test-image.jpg', { type: 'image/jpeg' }),
        url: 'test-image-url',
        id: '1'
      }
    ];
    mount(<MainWorkArea uploadedImageFiles={mockUploadedFiles} currentView="upload" />);
    cy.get('.image-display').find('img').should('be.visible');
    cy.contains('test-image.jpg');
  });

  it('displays delete button for uploaded images', () => {
    const mockUploadedFiles = [
      {
        file: new File([''], 'test-image.jpg', { type: 'image/jpeg' }),
        url: 'test-image-url',
        id: '1'
      }
    ];
    const onDeleteImage = cy.spy().as('deleteImageSpy');
    mount(
        <MainWorkArea
            uploadedImageFiles={mockUploadedFiles}
            currentView="upload"
            onDeleteImage={onDeleteImage}
        />
    );
    cy.get('[aria-label="delete image"]').click();
    cy.get('@deleteImageSpy').should('have.been.calledOnce');
  });

  it('switches between binary and color display modes', () => {
    const mockUploadedFiles = [
      {
        file: new File([''], 'test-image.jpg', { type: 'image/jpeg' }),
        url: 'test-image-url',
        id: '1'
      }
    ];
    const mockPredictionResults = [
      {
        binaryResult: 'binary-image-url',
        colorResult: 'color-image-url',
        processingTime: '2023-07-31T12:00:00Z',
        confidence: 0.95
      }
    ];
    mount(
        <MainWorkArea
            uploadedImageFiles={mockUploadedFiles}
            predictionResults={mockPredictionResults}
            currentView="results"
            displayModes={{0: 'binary'}}
            setDisplayModes={cy.spy().as('setDisplayModesSpy')}
        />
    );
    cy.contains('Processed Image (binary)');
    cy.get('[aria-label="switch to the other image"]').click();
    cy.get('@setDisplayModesSpy').should('have.been.calledOnce');
  });

  it('displays loading state while processing', () => {
    const mockUploadedFiles = [
      {
        file: new File([''], 'test-image.jpg', { type: 'image/jpeg' }),
        url: 'test-image-url',
        id: '1'
      }
    ];
    const mockPredictionResults = [null]; // null represents a processing state
    mount(
        <MainWorkArea
            uploadedImageFiles={mockUploadedFiles}
            predictionResults={mockPredictionResults}
            currentView="results"
        />
    );
    cy.get('.processing-placeholder').should('be.visible');
    cy.contains('Processing...');
  });
});