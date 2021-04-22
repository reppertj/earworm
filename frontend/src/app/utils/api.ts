import axios from 'axios';

const CancelToken = axios.CancelToken;
export const cancelTokenSource = CancelToken.source();

const backendAxios = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
  cancelToken: cancelTokenSource.token,
});

export const postEmbeddingsSearch = ({ queryKey }) => {
  const [_key, embeddings, page] = queryKey;
  const data = {
    k: (page + 1) * 50,
    embeddings,
  };
  return backendAxios.post('/embeddings/search', data);
};
