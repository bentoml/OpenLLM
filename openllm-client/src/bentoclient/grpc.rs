use crate::pb::bentoml::grpc::v1::bento_service_client::BentoServiceClient;
use crate::pb::bentoml::grpc::v1::*;
use tonic::transport::{Channel, Uri};

#[derive(Debug, Clone)]
pub struct GrpcClient {
    inner: BentoServiceClient<Channel>,
}

impl GrpcClient {
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;
        Ok(Self { inner: BentoServiceClient::new(channel) })
    }

    pub async fn service_metadata(&mut self) -> Result<(String, Vec<service_metadata_response::InferenceApi>)> {
        let request = tonic::Request::new(ServiceMetadataRequest {});
        let response = self.inner.service_metadata(request).await?.into_inner();
        Ok((response.name, response.apis))
    }

    pub async fn call<T>(&mut self, api_name: &str, data: T) -> Result<tonic::Response<Response>> {
        let request = RequestBuilder::build_req(api_name, data).await?;
        let response = self.inner.call(request).await?.into_inner();
        Ok(response)
    }
}

// TODO: Support NdArray, File, Multipart, DataFrame, Json, SerializedBytes, Series
trait RequestBuilder<T> {
    pub async fn build_req(&self, api_name: &str, data: T) -> Result<tonic::Request>;
}

impl RequestBuilder<String> for String {
    async fn build_req(&self, api_name: &str, data: &str) -> Result<tonic::Request> {
    let request = tonic::Request::new(Request {
        api_name: api_name.to_string(),
        content: Some(request::Content::Text(data.to_string())),
    });
    Ok(request)
    }
}
