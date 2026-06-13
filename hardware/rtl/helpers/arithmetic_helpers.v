// Functional helper modules used by structural approximate multiplier
// netlists. During synthesis, these modules are flattened and remapped
// to cells from the selected Liberty standard-cell library.

module FAX1(input A, input B, input C, output YS, output YC);
  assign YS = A ^ B ^ C;
  assign YC = (A & B) | (A & C) | (B & C);
endmodule

module HAX1(input A, input B, output YS, output YC);
  assign YS = A ^ B;
  assign YC = A & B;
endmodule
