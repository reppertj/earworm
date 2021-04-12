const getAuthorizeHref = (authEndpoint: string, scopes: string[]): string => {
  const clientId = process.env.REACT_APP_SPOTIFY_CLIENT_ID;
  const redirectUri = process.env.REACT_APP_REDIRECT_URL;
  return `${authEndpoint}?client_id=${clientId}&redirect_uri=${redirectUri}&scope=${scopes.join(
    '%20',
  )}&response_type=token`;
};

function removeHashParamsFromUrl() {
  window.history.pushState(
    '',
    document.title,
    window.location.pathname + window.location.search,
  );
}

function generateRandomString(length: number) {
  let text = '';
  const possible =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < length; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

interface HashParams {
  [key: string]: string;
}

function getHashParams(): HashParams {
  return window.location.hash
    .substring(1)
    .split('&')
    .reduce(function (initial: { [key: string]: any }, item) {
      if (item) {
        var parts = item.split('=');
        initial[parts[0]] = decodeURIComponent(parts[1]);
      }
      return initial;
    }, {});
}

export {
  getAuthorizeHref,
  generateRandomString,
  getHashParams,
  removeHashParamsFromUrl,
};
export type { HashParams };
