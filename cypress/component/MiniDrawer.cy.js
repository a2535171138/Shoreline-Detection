/// <reference types="cypress" />
import React from 'react';
import { mount } from 'cypress/react18';
import { BrowserRouter as Router } from 'react-router-dom';
import MiniDrawer from '../../src/components/MiniDrawer';

describe('MiniDrawer Component', () => {
    const createDefaultProps = () => ({
        onFileUpload: cy.stub().as('onFileUpload'),
        onClearImages: cy.stub().as('onClearImages'),
        onGetResult: cy.stub().as('onGetResult'),
        onToggleAllDisplayModes: cy.stub().as('onToggleAllDisplayModes'),
        onDownloadAll: cy.stub().as('onDownloadAll'),
        onToggleQualityCheck: cy.stub().as('onToggleQualityCheck'),
        qualityCheckEnabled: false,
        hasResults: false,
        onSwitchView: cy.stub().as('onSwitchView'),
        currentView: 'upload',
        onViewLogs: cy.stub().as('onViewLogs'),
    });

    const mountComponent = (props = {}) => {
        mount(
            <Router>
                <MiniDrawer {...createDefaultProps()} {...props}>
                    <div>Test Content</div>
                </MiniDrawer>
            </Router>
        );
    };

    it('renders the drawer and app bar', () => {
        mountComponent();
        cy.get('.MuiDrawer-root').should('exist');
        cy.get('.MuiAppBar-root').should('exist');
    });


    it('toggles quality check when switch is clicked', () => {
        mountComponent();
        cy.get('.MuiSwitch-root').eq(1).click();
        cy.get('@onToggleQualityCheck').should('have.been.called');
    });

    it('switches view when view switch is toggled', () => {
        mountComponent();
        cy.get('.MuiSwitch-root').eq(0).click();
        cy.get('@onSwitchView').should('have.been.called');
    });

    it('opens user guide when account icon is clicked', () => {
        mountComponent();
        cy.get('button[aria-label="user guide"]').click();
        cy.contains('User Guide').click();
        cy.url().should('include', '/user-guide');
    });

    it('displays correct title', () => {
        mountComponent();
        cy.get('.MuiTypography-root').contains('AI-Driven Shoreline Detection');
    });


    it('disables quality check switch when qualityCheckEnabled is false', () => {
        mountComponent({ qualityCheckEnabled: false });
        cy.get('.MuiSwitch-root').eq(1).should('not.have.class', 'Mui-checked');
    });

});