"use client"

import { useState, useEffect } from "react";
import { useGlobalContext } from "../context/GlobalContext";

export default function Home() {
	const { images, setImages, boxedImages, setBoxedImages, isbns, setIsbns } = useGlobalContext();
	const [isbnInput, setIsbnInput] = useState('');

	const handleInputChange = (e) => {
        setIsbnInput(e.target.value);
    };

	const addISBN = () => {
		setIsbns(isbns => [...isbns, isbnInput]);
		setIsbnInput('')
	}

	const removeISBN = (index: number) => {
        setIsbns(isbns.slice(0, index).concat(isbns.slice(index + 1)));
    };

    return (
		<main className="flex">
			<div className="w-3/4 border-r border-gray-300 p-4">
				<h1>Images</h1>
				<div className="image-container flex flex-wrap justify-start items-start">
                    {boxedImages.map((image, index) => (
						<div className="flex flex-wrap justify-center items-center min-w-0 p-2">
                        	<img key={index} src={image} alt="img" className="max-w-xs h-auto" />
						</div>
                    ))}
                </div>
			</div>
			<div className="w-1/4 p-4">
				<h1>ISBNS</h1>
				{isbns.map((isbn, index) => (
					<div key={index} className="flex items-center space-x-2 p-2">
						<span className="flex-1">{isbn}</span>
						<button onClick={() => removeISBN(index)} className="px-2 py-1 bg-red-500 text-white rounded">X</button>
					</div>
            	))}
				<div className="flex items-center space-x-2 p-2">
					<input type="text" value={isbnInput} onChange={handleInputChange}
						className="flex-1 p-2 border border-gray-300 rounded text-black" placeholder="New ISBN" />
					<button onClick={addISBN} className="px-2 py-1 bg-green-500 rounded hover:bg-green-600
							active:bg-green-700 focus:outline-none focus:ring focus:ring-green-300">✓</button>
				</div>
			</div>
		</main>
    );
  }