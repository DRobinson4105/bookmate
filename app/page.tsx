import React from 'react';
import ImagesUpload from './components/ImagesUpload';


export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <ImagesUpload />
    </main>
  );
}
