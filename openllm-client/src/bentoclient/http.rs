use reqwest::{Client, Url};

#[derive(Debug, Clone)]
pub struct HttpClient {
    inner: Client,
    url: Url,
}

impl HttpClient {
    pub async fn connect(uri: str) -> Result<Self> {
        Ok(Self { inner: Client::new(), url: Url::parse(uri)? })
    }

    pub async fn service_metadata(&mut self) -> Result<(String, Vec<HashMap<String, T>>)> {
        let response = self.inner.get(self.url.join("docs.json")).json().await?;
        Ok((response["info"]["title"], response["paths"]))
    }
}
