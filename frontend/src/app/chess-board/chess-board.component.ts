import { Component, OnInit, ElementRef } from '@angular/core';
import { piecesDict } from './piecesDict';
import { pieceValueDict } from './pieceValueDict';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

/* Enable Jquery */
// declare var $: any;

@Component({
  selector: 'chess-board',
  templateUrl: './chess-board.component.html',
  styleUrls: ['./chess-board.component.css']
})


export class ChessBoardComponent implements OnInit {

  constructor(private http: HttpClient) { 
    // fetch("${this.server_root_url}/get_env")
    //      .then(response => response.json())
    //      .then(this.loadBoard);
  }

  ngOnInit(): void {
    // fetch("${this.server_root_url}/get_env")
    //      .then(response => response.json())
    //      .then(this.loadBoard);

    this.state[0] = [-3, -5, -4, -2, -1, -4, -5, -3];
    this.state[1] = Array(8).fill(-6);
    this.state[6] = Array(8).fill(6);
    this.state[7] = [3, 5, 4, 2, 1, 4, 5, 3];

    this.awaitingResponse = true;

    this.http.get(`${this.server_root_url}/get_env`, {withCredentials: true})
      .subscribe(this.loadBoard);
  }

  static blackRook = "<div class='black-piece'>♜</div>";
  static blackKnight = "<div class='black-piece'>♞</div>";
  static blackBishop = "<div class='black-piece'>♝</div>";
  static blackQueen = "<div class='black-piece'>♛</div>";
  static blackKing = "<div class='black-piece'>♚</div>";
  static blackPawn = "<div class='black-piece'>♟</div>";

  static whiteRook = "<div class='white-piece'>♖</div>";
  static whiteKnight = "<div class='white-piece'>♘</div>";
  static whiteBishop = "<div class='white-piece'>♗</div>";
  static whiteQueen = "<div class='white-piece'>♕</div>";
  static whiteKing = "<div class='white-piece'>♔</div>";
  static whitePawn = "<div class='white-piece'>♙</div>";

  static highlightSquare = "<div class='highlight-square'></div>";

  static pieceToValue:pieceValueDict = {
    6: 'PAWN',
    5: 'KNIGHT',
    4: 'BISHOP',
    3: 'ROOK',
    2: 'QUEEN',
    1: 'KING',
  }

  static pieces:piecesDict = {
    '-6': ChessBoardComponent.blackPawn,
    '-5': ChessBoardComponent.blackKnight,
    "-4": ChessBoardComponent.blackBishop,
    "-3": ChessBoardComponent.blackRook,
    "-2": ChessBoardComponent.blackQueen,
    "-1": ChessBoardComponent.blackKing,
    "6": ChessBoardComponent.whitePawn,
    "5": ChessBoardComponent.whiteKnight,
    "4": ChessBoardComponent.whiteBishop,
    "3": ChessBoardComponent.whiteRook,
    "2": ChessBoardComponent.whiteQueen,
    "1": ChessBoardComponent.whiteKing,
    "0": ''
  }

  server_root_url = "http://localhost:5000";
  //server_root_url = "https://daya-chess.wl.r.appspot.com";

  possible_moves:Array<any> = [];
  state:Array<Array<number>> = Array(8).fill(null).map(()=>Array(8).fill(0));
  done = false;
  result = 0;
  awaitingResponse = true;

  color = "WHITE";

  selected = -1;
  promotionEnabled = false;
  promotionFile = 0;

  files = ["a", "b", "c", "d", "e", "f", "g", "h"]

  moveIndex = -1;    // used to save move for promotions

  getColorAsInt = ():number => {
    return this.color === 'WHITE' ? 1 : -1;
  }

  getPiece = (value:number, highlightable:boolean) => {
    return (highlightable ? ChessBoardComponent.highlightSquare : "") + ChessBoardComponent.pieces[value.toString() as keyof piecesDict];
  }

  convertIndex = (index:number) => {
    return this.color === 'WHITE' ? index : 7 - index;
  }

  updateFiles = () => {
    if ((this.color === 'WHITE' && this.files[0] === "h") || (this.color === 'BLACK' && this.files[0] == "a")) {
      this.files.reverse();
    }
  }

  loadBoard = (data:any) => {
    console.log(data);
    this.possible_moves = data.possible_moves;
    this.state = data.state;
    this.color = data.color;
    this.updateFiles();
    this.awaitingResponse = false;
  }

  updateBoard = (data:any, showMove:boolean=true) => {
    console.log(data);
    this.state = data.state;
    if ("result" in data) {
      this.done = true;
      this.result = Math.sign(data.result);
      this.possible_moves = [];
    }
    else
      this.possible_moves = data.possible_moves;
    
    this.awaitingResponse = false;
    
  }

  displayPromotionModal = (file:number) => {
    //let row:number = color == 'WHITE' ? 0 : 7;
    this.promotionEnabled = true;
    this.promotionFile = file;
  }

  promote = (pieceIndex:number) => {
    if (this.moveIndex == -1)
      return;

    let row = this.color == 'WHITE' ? 0 : 7;
    this.state[row][this.promotionFile] = pieceIndex * this.getColorAsInt();
    this.promotionEnabled = false;
    this.promotionFile = 0;


    let requestOptions = {
      withCredentials: true
    }

    let piece = ChessBoardComponent.pieceToValue[pieceIndex as keyof pieceValueDict];
    // Send move to backend
    this.http.put(`${this.server_root_url}/move/${this.moveIndex}?promote=${piece}`, null, requestOptions)
            .subscribe(this.updateBoard);

    // Reset moveIndex
    this.moveIndex = -1;
  }

  onClick = (row:number, index:number) => {

    /* If we click on our own piece, we select it */
    if ((this.state[8 - row][index] > 0 && this.color == 'WHITE') || (this.state[8 - row][index] < 0 && this.color == 'BLACK')) {
      if (this.selected === row * 8 + index)
        this.selected = -1
      else
        this.selected = row * 8 + index;
    }

    /* If something is selected, we can make a move or unselect */
    else if (this.selected !== -1) {
      /* Unselect */
      if (this.selected === row * 8 + index)
        this.selected = -1;
      
      /* Otherwise, we are trying to make a move. 
      ** 1. Verify we are not waiting for a response from the server
      ** 2. Verify move is legal
      ** 3. Change state to reflect move
      */
      
      else {
        if (this.awaitingResponse)
          return;
        let idx = -1;

        let startY = 8 - (Math.floor(this.selected / 8)), startX = this.selected % 8;
        let endY = 8 - row, endX = index;
        if (this.possible_moves.some(move => {
          return move[0][0] === startY && move[0][1] === startX && move[1][0] === endY && move[1][1] === endX;
        })) {
          idx = this.possible_moves.findIndex(move => {
            return move[0][0] === startY && move[0][1] === startX && move[1][0] === endY && move[1][1] === endX;
          });
          
          console.log(startX, startY, endX, endY);
          this.state[endY][endX] = this.state[startY][startX];
          this.state[startY][startX] = 0;
          //TODO: Implement en-passant capture on client

        }
        else {
          /* Consider special cases 
          ** 1. Castling
          */
          if (startX == 4 && startY == 7 && endY == 7) {
            if (endX == 2 && this.possible_moves.includes("CASTLE_QUEEN_SIDE_WHITE")) {
              this.state[7][4] = 0;
              this.state[7][0] = 0;
              this.state[7][2] = 1;
              this.state[7][3] = 3;
              idx = this.possible_moves.indexOf("CASTLE_QUEEN_SIDE_WHITE");
            }
            else if (endX == 6 && this.possible_moves.includes("CASTLE_KING_SIDE_WHITE")) {
              this.state[7][4] = 0;
              this.state[7][7] = 0;
              this.state[7][6] = 1;
              this.state[7][5] = 3;
              idx = this.possible_moves.indexOf("CASTLE_KING_SIDE_WHITE");
            }
          }
          else if (startX == 4 && startY == 0 && endY == 0) {
            if (endX == 2 && this.possible_moves.includes("CASTLE_QUEEN_SIDE_BLACK")) {
              this.state[0][4] = 0;
              this.state[0][0] = 0;
              this.state[0][2] = -1;
              this.state[0][3] = -3;
              idx = this.possible_moves.indexOf("CASTLE_QUEEN_SIDE_BLACK");
            }
            else if (endX == 6 && this.possible_moves.includes("CASTLE_KING_SIDE_BLACK")) {
              this.state[0][4] = 0;
              this.state[0][7] = 0;
              this.state[0][6] = -1;
              this.state[0][5] = -3;
              idx = this.possible_moves.indexOf("CASTLE_KING_SIDE_BLACK");
            }
          }
        }

        if (idx >= 0) {
          /* Get new state (including opponent's move) */
          this.selected = -1;

          /* Check for promotion possibility */
          if ((endY == 0 && this.state[endY][endX] == 6) || (endY == 7 && this.state[endY][endX] == -6)) {
            this.displayPromotionModal(endX);
            this.moveIndex = idx;
            return;
          }
          
          /* Disable Moving */
          this.awaitingResponse = true;

          let requestOptions = {
            withCredentials: true
          }

          this.http.put(`${this.server_root_url}/move/${idx}`, null, requestOptions)
            .subscribe(this.updateBoard);
        }
      }
    }
  }

  startNewGame = (color:string) => {
    this.color = color;
    this.updateFiles();
    this.http.get(`${this.server_root_url}/new_env?color=${color}`,  { withCredentials: true })
      .subscribe(data => this.updateBoard(data, false));
  }


}
