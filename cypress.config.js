const { defineConfig } = require("cypress");

module.exports = defineConfig({
  e2e: {
    framework: 'react',
    bundler: 'webpack',
    specPattern: "cypress/e2e/**/*.cy.{js,jsx,ts,tsx}", // 确保测试文件的匹配模式
    testIsolation: false,
    supportFile: "cypress/support/e2e.js",
    baseUrl: 'http://localhost:3000',  
  },

  component: {
    devServer: {
      framework: "create-react-app",
      bundler: "webpack",
    },
    supportFile: 'cypress/support/component.js',
    indexHtmlFile: 'cypress/support/component-index.html',
  },
});

