describe('API Tests', () => {
  const baseUrl = 'http://localhost:5000';
  const scene = 'Narrabeen';

  it('POST /predict/:scene should return a prediction', () => {
    cy.fixture('image1.jpg', 'binary')
      .then(Cypress.Blob.binaryStringToBlob)
      .then((fileContent) => {
        const formData = new FormData();
        const testFile = new File([fileContent], 'image1.jpg', { type: 'image/jpeg' });
        formData.append('file', testFile);

        cy.request({
          method: 'POST',
          url: `${baseUrl}/predict/${scene}`,
          body: formData,
          headers: {
            'Content-Type': 'multipart/form-data'
          },
        }).then((response) => {
          expect(response.status).to.eq(200);
        });
      });
  });

  it('POST /predict/:scene should return error for missing file', () => {
    cy.request({
      method: 'POST',
      url: `${baseUrl}/predict/${scene}`,
      failOnStatusCode: false,
    }).then(response => {
      expect(response.status).to.eq(400);
      expect(response.body).to.have.property('error', 'No file part');
    });
  });

  it('POST /predict/:scene should return error for no selected file', () => {
    // 创建一个空的文件上传
    const formData = new FormData();
    const emptyFile = new File([''], '', { type: 'text/plain' });
    formData.append('file', emptyFile);

    cy.request({
      method: 'POST',
      url: `${baseUrl}/predict/${scene}`,
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      failOnStatusCode: false,
    }).then(response => {
      expect(response.status).to.eq(400);
      // expect(response.body).to.have.property('error', 'No selected file');
    });
  });

  it('POST /predict/:scene should return error for invalid file type', () => {
    cy.fixture('invalid.txt', 'binary')
      .then(Cypress.Blob.binaryStringToBlob)
      .then((fileContent) => {
        const formData = new FormData();
        formData.append('file', new Blob([fileContent], { type: 'text/plain' }), 'invalid.txt');

        cy.request({
          method: 'POST',
          url: `${baseUrl}/predict/${scene}`,
          body: formData,
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          failOnStatusCode: false
        }).then((response) => {
          expect(response.status).to.eq(400);
          // expect(response.body).to.have.property('error', 'Invalid file type');
        });
      });
  });

  it('GET /download_all/pixels should download a CSV file with pixel data', () => {
    cy.request({
      method: 'GET',
      url: `${baseUrl}/download_all/pixels`,
    }).then(response => {
      expect(response.status).to.eq(200);
      expect(response.headers['content-type']).to.eq('text/csv; charset=utf-8');
    });
  });

  it('GET /download_all/binary should download a zip file with binary images', () => {
    cy.request({
      method: 'GET',
      url: `${baseUrl}/download_all/binary`,
    }).then(response => {
      expect(response.status).to.eq(200);
      expect(response.headers['content-type']).to.eq('application/zip');
    });
  });

  it('GET /download_all/color should download a zip file with color images', () => {
    cy.request({
      method: 'GET',
      url: `${baseUrl}/download_all/color`,
    }).then(response => {
      expect(response.status).to.eq(200);
      expect(response.headers['content-type']).to.eq('application/zip');
    });
  });

  it('GET /download_all/all should download a zip file with all results', () => {
    cy.request({
      method: 'GET',
      url: `${baseUrl}/download_all/all`,
    }).then(response => {
      expect(response.status).to.eq(200);
      expect(response.headers['content-type']).to.eq('application/zip');
    });
  });
});
