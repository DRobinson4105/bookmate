"use client";
import React, { useState, ChangeEvent } from 'react';
import { useGlobalContext } from './context/GlobalContext';
import Link from 'next/link';
import axios from 'axios';

export default function Home() {
	const { images, setImages, boxedImages, setBoxedImages, isbns, setIsbns } = useGlobalContext();
    const [nonImageFiles, setNonImageFiles] = useState<boolean>(false);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            const files = Array.from(event.target.files);
            const filteredImages = files.filter(file => {
                const isImage = file.type.startsWith('image/');
				setNonImageFiles(!isImage)
                return isImage;
            });
            setImages(prevImages => [...prevImages, ...filteredImages]);
        }
    };

    const removeFile = (index: number) => {
        setImages(images.slice(0, index).concat(images.slice(index + 1)));
    };

	const handleDone = async () => {
		let images0 = [], isbns0 = [];

		for (let image of images) {
			let formData = new FormData();
			formData.append("file", image as Blob);
			const response = await axios.post(
				"http://127.0.0.1:5328/api/getISBNs",
				formData,
				{
					headers: {
						"Content-Type": "multipart/form-data",
					},
				}
			);

			const data = await response.data;
            images0.push(data.image)
            for (let isbn of data.isbns)
                isbns0.push(isbn)
		}

		setBoxedImages(images0)
        setIsbns(isbns0)
	}

  return (
	<main className="flex min-h-screen flex-col items-center justify-between p-24">
              <div className="flex flex-col justify-center items-center ">
            <div>
            <h1 className="m-10 text-7xl">Upload Images</h1>
            {images.map((file, index) => (
                <div key={index} className="flex items-center space-x-2 p-2">
                    <span className="flex-1">{file.name}</span>
                    <button onClick={() => removeFile(index)} className="px-2 py-1 bg-red-500 text-white rounded">X</button>
                </div>
            ))}
            </div>
            <input type="file" onChange={handleFileChange} className="
                block text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0
                file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-300"/>
            <Link onClick={handleDone} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-300" href={"/isbn-review"}>Done</Link>
			{nonImageFiles && <p className="text-red-500">File is not an image</p>}
        </div>
    </main>
  );
}