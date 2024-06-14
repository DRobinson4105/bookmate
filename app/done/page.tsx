"use client"

import { useState, useEffect } from "react";
import { useGlobalContext } from "../context/GlobalContext";
import Link from 'next/link';
import axios from 'axios';
import { useRouter } from 'next/navigation';

export default function Home() {
	const { images, setImages, boxedImages, setBoxedImages, isbns, setIsbns, spreadsheet, setSpreadsheet } = useGlobalContext();

    return (
		<main className="flex">
			<div className="w-1/4 p-4">
                <a href={spreadsheet} download="spreadsheet.xlsx" className="
                mt-4 px-4 py-2 bg-blue-500 text-white rounded 
                shadow hover:bg-blue-300">Download</a>
			</div>
		</main>
    );
  }