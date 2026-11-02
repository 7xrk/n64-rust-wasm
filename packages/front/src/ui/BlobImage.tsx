/** An image of a Blob through an object URL that lives as long as the element shows that Blob. */

import { useEffect, useState } from 'preact/hooks';

interface Props {
  blob: Blob;
  width: number;
  height: number;
}

export function BlobImage({ blob, width, height }: Props) {
  const [url, setUrl] = useState<string>();

  useEffect(() => {
    const next = URL.createObjectURL(blob);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob]);

  return <img src={url} alt="" width={width} height={height} />;
}
