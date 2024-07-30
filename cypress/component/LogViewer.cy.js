/// <reference types="cypress" />
import React from 'react';
import { mount } from 'cypress/react18';
import LogViewer from '../../src/components/LogViewer';
import { LogProvider } from '../../src/LogContext';

describe('LogViewer Component', () => {
    const mockLogs = [
        { message: 'Test log 1', timestamp: '2023-07-31T12:00:00Z', severity: 'info' },
        { message: 'Test log 2', timestamp: '2023-07-31T12:01:00Z', severity: 'warning' },
        { message: 'Test log 3', timestamp: '2023-07-31T12:02:00Z', severity: 'error' },
    ];

    const mountComponent = (props = {}) => {
        const defaultProps = {
            open: true,
            onClose: cy.stub().as('onClose'),
        };

        return mount(
            <LogProvider value={{ getAllLogs: () => mockLogs }}>
                <LogViewer {...defaultProps} {...props} />
            </LogProvider>
        );
    };

    it('renders correctly when open', () => {
        mountComponent();
        cy.get('.MuiDialog-root').should('be.visible');
        cy.contains('Operation Logs').should('be.visible');
    });

    it('does not render when closed', () => {
        mountComponent({ open: false });
        cy.get('.MuiDialog-root').should('not.exist');
    });



    it('calls onClose when close button is clicked', () => {
        mountComponent();
        cy.get('button[aria-label="close"]').click();
        cy.get('@onClose').should('have.been.calledOnce');
    });


    it('has accessible close button', () => {
        mountComponent();
        cy.get('button[aria-label="close"]')
            .should('have.attr', 'aria-label', 'close')
            .and('be.visible');
    });


});