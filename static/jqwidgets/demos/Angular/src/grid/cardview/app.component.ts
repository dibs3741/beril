﻿import { AfterViewInit, Component, OnInit, ViewChild, ViewEncapsulation } from '@angular/core';
import { jqxGridComponent } from 'jqwidgets-ng/jqxgrid';
import { jqxButtonComponent } from 'jqwidgets-ng/jqxbuttons';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  encapsulation: ViewEncapsulation.None
})
export class AppComponent implements AfterViewInit, OnInit {
  @ViewChild('grid', { static: false }) grid: jqxGridComponent;
  @ViewChild('button', { static: false }) button: jqxButtonComponent;

	columns = [
		{ text: 'Name', datafield: 'firstname', width: 120 },
		{ text: 'Last Name', datafield: 'lastname', width: 120 },
		{ text: 'Product', datafield: 'productname', width: 180 },
		{ text: 'Quantity', datafield: 'quantity', width: 80, cellsalign: 'right' },
		{ text: 'Unit Price', datafield: 'price', width: 90, cellsalign: 'right', cellsformat: 'c2' },
		{ text: 'Total', datafield: 'total', cellsalign: 'right', cellsformat: 'c2' }
	];
				
	cardviewcolumns = [
		{
			width: 'auto',
			datafield: 'firstname'
		},
		{
			width: 'auto',
			datafield: 'lastname'
		},
		{
			width: 300,
			datafield: 'productname'
		}
	]

  generateData(): any[] {
        let data = new Array();

        let firstNames =
            [
                'Andrew', 'Nancy', 'Shelley', 'Regina', 'Yoshi', 'Antoni', 'Mayumi', 'Ian', 'Peter', 'Lars', 'Petra', 'Martin', 'Sven', 'Elio', 'Beate', 'Cheryl', 'Michael', 'Guylene'
            ];

        let lastNames =
            [
                'Fuller', 'Davolio', 'Burke', 'Murphy', 'Nagase', 'Saavedra', 'Ohno', 'Devling', 'Wilson', 'Peterson', 'Winkler', 'Bein', 'Petersen', 'Rossi', 'Vileid', 'Saylor', 'Bjorn', 'Nodier'
            ];

        let productNames =
            [
                'Black Tea', 'Green Tea', 'Caffe Espresso', 'Doubleshot Espresso', 'Caffe Latte', 'White Chocolate Mocha', 'Cramel Latte', 'Caffe Americano', 'Cappuccino', 'Espresso Truffle', 'Espresso con Panna', 'Peppermint Mocha Twist'
            ];

        let priceValues =
            [
                '2.25', '1.5', '3.0', '3.3', '4.5', '3.6', '3.8', '2.5', '5.0', '1.75', '3.25', '4.0'
            ];

        for (let i = 0; i < 200; i++) {
            let row = {};
            let productindex = Math.floor(Math.random() * productNames.length);
            let price = parseFloat(priceValues[productindex]);
            let quantity = 1 + Math.round(Math.random() * 10);
            row['firstname'] = firstNames[Math.floor(Math.random() * firstNames.length)];
            row['lastname'] = lastNames[Math.floor(Math.random() * lastNames.length)];
            row['productname'] = productNames[productindex];
            row['price'] = price;
            row['quantity'] = quantity;
            row['total'] = price * quantity;
            data[i] = row;
        }

        return data;
    }
	
  source = new jqx.dataAdapter({
		localData: this.generateData(),
		datatype: 'array',
        datafields:
        [
            { name: 'firstname', type: 'string' },
            { name: 'lastname', type: 'string' },
            { name: 'productname', type: 'string' },
            { name: 'quantity', type: 'number' },
            { name: 'price', type: 'number' },
            { name: 'total', type: 'number' }
        ]
   });
	 
  ngOnInit() {
  }

  switchBtnOnClick(): void {
		var cardview = this.grid.cardview();
		this.grid.cardview(!cardview);
  };
	
  ngAfterViewInit() {
	
  }
}