import cmath as cm
from typing import List, Union, Tuple

############################# Defining Types ############################

BIT_SEQUENCE_TYPE = List[int]
SYMBOL_SEQUENCE_TYPE = List[complex]
BIT_TO_SYMBOL_MAP_TYPE = List[ List[ Union[ complex, List[int] ] ] ]
SYMBOL_BLOCKS_TYPE = List[ List[complex] ]
CHANNEL_IMPULSE_RESPONSE_TYPE = List[complex]
RANDOM_VALUES_SYMBOLS_TYPE = List[ List[ List[float] ] ]
RANDOM_VALUES_CIR_TYPE = List[ List[ List[float] ] ]
NOISY_SYMBOL_SEQUENCE_TYPE = List[ List[complex] ]
SER_TYPE = Union[float, None]
BER_TYPE = Union[float, None]

#########################################################################
#                   Given Modulation Bit to Symbol Maps                 #
#########################################################################

MAP_BPSK : BIT_TO_SYMBOL_MAP_TYPE = [
    [(-1 + 0j), [0]],
    [(1 + 0j), [1]],
]

MAP_4QAM : BIT_TO_SYMBOL_MAP_TYPE = [
    [( 1 + 1j)/cm.sqrt(2), [0, 0]],
    [(-1 + 1j)/cm.sqrt(2), [0, 1]],
    [(-1 - 1j)/cm.sqrt(2), [1, 1]],
    [( 1 - 1j)/cm.sqrt(2), [1, 0]],
]

#########################################################################
#                           Evaluation Function                         #
#########################################################################

def evaluate():
    """
    Your code used to evaluate your system should be written here.
             !!! NOTE: This function will not be marked !!!
    """

    # assist_bit_to_symbol
    bitBPSK = [0, 1, 0, 1]
    symbolSequenceBPSK = assist_bit_to_symbol(bitBPSK, MAP_BPSK)
    #print(symbolSequenceBPSK)

    bit4QAM = [0, 1, 1, 0, 0, 0, 1, 1]
    symbolSequence4QAM = assist_bit_to_symbol(bit4QAM, MAP_4QAM)
    #print(symbolSequence4QAM)

    # assist_symbol_to_bit
    bitsBPSK = assist_symbol_to_bit(symbolSequenceBPSK, MAP_BPSK)
    #print(bitsBPSK)
    bits4QAM = assist_symbol_to_bit(symbolSequence4QAM, MAP_4QAM)
    #print(bits4QAM)

    #assist_split_symbols_into_blocks
    blocked = assist_split_symbols_into_blocks(symbolSequence4QAM,2)
    #print(blocked)

    #assist_combine_blocks_into_symbols
    combined = assist_combine_blocks_into_symbols(blocked)
    #print(combined)

    #DFE_BPSK_BLOCK
    channelImpulse = [0.1685, -0.3112, -0.6987]
    recivedImpulse = [-0.7442, -1.0812, -0.4788, 1.2556]
    transmittedBPSK = DFE_BPSK_BLOCK(recivedImpulse,channelImpulse)  # check with symbolSequenceBPSK as well
    #print(transmittedBPSK)

    #DFE_4QAM_BLOCK
    transmitted4QAM = DFE_4QAM_BLOCK(symbolSequence4QAM,channelImpulse)
    #print(transmitted4QAM)

    #MLSE_BPSK_BLOCK
    c = [0.5824, -0.7065, 0.3645]
    r = [-0.1967, -0.4586, 0.7886, 1.2101, -0.1460, 0.271]
    transmitted_sequence_BPSK = MLSE_BPSK_BLOCK(r, c)
    #print(transmitted_sequence_BPSK)

    #MLSE_4QAM_BLOCK
    transmitted_sequence_4QAM = MLSE_4QAM_BLOCK(r,c)
    #print(transmitted_sequence_4QAM)

    #SER_BER_BPSK_DFE_STATIC
    bit_sequence = [0, 1, 1, 0, 1, 0, 0, 1]
    block_size = 4
    random_values_for_symbols = [
        [[0.01, 0], [0.02, 0], [0.03, 0], [0.04, 0]],  # Block 1
        [[0.05, 0], [0.06, 0], [0.07, 0], [0.08, 0]]  # Block 2
    ]

    snr = 10
    noisy_symbol_blocks_DFE_static_bpsk, SER, BER = SER_BER_BPSK_DFE_STATIC(bit_sequence, block_size, random_values_for_symbols, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_DFE_static_bpsk)
    # print("SER:", SER)
    # print("BER:", BER)

    #SER_BER_BPSK_MLSE_STATIC
    noisy_symbol_blocks_MLSE_static_bpsk, SER, BER = SER_BER_BPSK_MLSE_STATIC(bit_sequence, block_size,random_values_for_symbols, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_MLSE_static_bpsk)
    # print("SER:", SER)
    # print("BER:", BER)

    #SER_BER_BPSK_DFE_DYNAMIC
    random_values_for_CIR = [
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # Block 1
        [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]  # Block 2
    ]

    noisy_symbol_blocks_DFE_dynamic_bpsk, SER, BER = SER_BER_BPSK_DFE_DYNAMIC(bit_sequence, block_size, random_values_for_symbols,random_values_for_CIR, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_DFE_dynamic_bpsk)
    # print("SER:", SER)
    # print("BER:", BER)

    # SER_BER_MLSE_DYNAMIC
    noisy_symbol_blocks_MLSE_dynamic_bpsk, SER, BER = SER_BER_BPSK_MLSE_DYNAMIC(bit_sequence, block_size,random_values_for_symbols,random_values_for_CIR, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_MLSE_dynamic_bpsk)
    # print("SER:", SER)
    # print("BER:", BER)

    #SER_BER_4QAM_DFE_STATIC
    noisy_symbol_blocks_DFE_static_4qam, SER, BER = SER_BER_4QAM_DFE_STATIC(bit_sequence, block_size,random_values_for_symbols, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_DFE_static_4qam)
    # print("SER:", SER)
    # print("BER:", BER)

    # SER_BER_4QAM_MLSE_STATIC
    noisy_symbol_blocks_MLSE_static_4qam, SER, BER = SER_BER_4QAM_MLSE_STATIC(bit_sequence, block_size,random_values_for_symbols, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_MLSE_static_4qam)
    # print("SER:", SER)
    # print("BER:", BER)

    #SER_BER_4QAM_DFE_DYNAMIC
    noisy_symbol_blocks_DFE_dynamic_4qam, SER, BER = SER_BER_4QAM_DFE_DYNAMIC(bit_sequence, block_size,random_values_for_symbols,random_values_for_CIR, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_DFE_dynamic_4qam)
    # print("SER:", SER)
    # print("BER:", BER)

    # SER_BER_MLSE_DYNAMIC
    noisy_symbol_blocks_MLSE_dynamic_4qam, SER, BER = SER_BER_4QAM_MLSE_DYNAMIC(bit_sequence, block_size,random_values_for_symbols,random_values_for_CIR, snr)
    # print("Noisy Symbol Blocks:", noisy_symbol_blocks_MLSE_dynamic_4qam)
    # print("SER:", SER)
    # print("BER:", BER)

    return
   
#########################################################################
#                           Assisting Functions                         #
#########################################################################

def assist_bit_to_symbol(bit_sequence: BIT_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Converts a sequence of bits to a sequence of symbols using the bit to symbol map.
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]   
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK and MAP_4QAM
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """
    symbol_sequence = []
    lenBitSequence = len(bit_to_symbol_map[0][1])  # Determine the length of the bit sequence for each symbol

    x = 0
    while x < len(bit_sequence):
        currBits = bit_sequence[x:x + lenBitSequence]  # extracts a chunk of bit_sequence from x to x+lenBit

        y = 0
        while y < len(bit_to_symbol_map):
            sym, mappedBits = bit_to_symbol_map[y]  # sym = complex num, mappedBits = corresponding bits
            if currBits == mappedBits:
                symbol_sequence.append(sym)
                break
            y += 1

        x += lenBitSequence

    return symbol_sequence

def assist_symbol_to_bit(symbol_sequence: SYMBOL_SEQUENCE_TYPE, bit_to_symbol_map: BIT_TO_SYMBOL_MAP_TYPE) -> BIT_SEQUENCE_TYPE:
    """
    Returns a sequence of bits that corresponds to the provided sequence of symbols containing noise using the bit to symbol map that respresent the modulation scheme and the euclidean distance
    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        bit_to_symbol_map -> type <class 'list'> : A list containing lists. Each list entry contains a complex number and a bit sequence, representing the symbol, and the corresponding bit sequence that maps to it.
                             the maps that will be used are given as MAP_BPSK and MAP_4QAM
          Example:
            [
                [ 1+1j,    [0, 0] ],
                [ -1+1j,   [0, 1] ],
                [ -1-1j,   [1, 1] ],
                [ 1-1j,    [1, 0] ]
            ]
    returns:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1] 

    """
    bit_sequence = []
    x = 0

    while x < len(symbol_sequence):
        symbol = symbol_sequence[x]
        minDist = float('inf')  # make mindist a large value
        closest = None  # used to store closest bit to complex num

        y = 0
        while y < len(bit_to_symbol_map):
            sym, mappedBits = bit_to_symbol_map[y]
            dist = abs(symbol - sym)

            if dist < minDist:
                minDist = dist
                closest = mappedBits

            y += 1

        if closest is not None:
            bit_sequence.extend(closest)

        x += 1

    return bit_sequence

def assist_split_symbols_into_blocks(symbol_sequence: SYMBOL_SEQUENCE_TYPE, block_size: int) -> SYMBOL_BLOCKS_TYPE:
    """
    Divides the given symbol sequence into blocks of length block_size, that the DFE and MLSE algorithm should be performed upon.
    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
    returns:
        symbol_blocks -> type <class 'list'> : List of lists. Each list entry should be a list representing a symbol sequence, which is a list containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
    """
    symbol_blocks = []
    x = 0

    while x < len(symbol_sequence):
        currBlock = symbol_sequence[x:x + block_size]
        symbol_blocks.append(currBlock)
        x += block_size

    return symbol_blocks

def assist_combine_blocks_into_symbols(symbol_blocks: SYMBOL_BLOCKS_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Combines the given blocks of symbol sequences into a single sequence of symbols.

    parameters:
        symbol_blocks -> type <class 'list'> : List of lists. Each list entry should be a list representing a symbol sequence, which is a list containing containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]

    returns:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """
    symbol_sequence = []
    x = 0

    while x < len(symbol_blocks):
        symbol_sequence.extend(symbol_blocks[x])
        x += 1

    return symbol_sequence

#########################################################################
#                         DFE and MLSE Functions                        #
#########################################################################

def DFE_BPSK_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the DFE algorithm on the given symbol sequence (which was modulated using the BPSK scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [1, 1]
    Only the transmitted data bits must be returned, thus exluding the prepended symbols. Thus len(symbol_sequence) equals len(transmitted_sequence).
    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]
    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """
    s = [1,1]
    symbols = [1, -1]
    c = impulse_response
    r = symbol_sequence
    h = 2
    x = 0

    while x < len(r):
        delta = []
        added = 0
        z = len(s) - 1
        w = 1

        while (z > -1) and (w < len(c)):
            added += (s[z] * c[w])
            w += 1
            z -= 1

        y = 0
        while y < len(symbols):
            dfe = abs(r[x] - (symbols[y] * c[0] + added)) ** 2
            delta.append(dfe)
            y += 1

        index = delta.index(min(delta))
        s.append(symbols[index])
        x += 1

    transmitted_sequence = s[h:]

    return transmitted_sequence

def DFE_4QAM_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the DFE algorithm on the given symbol sequence (which was modulated using the 4QAM scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]
    Only the transmitted data bits must be returned, thus exluding the prepended symbols. Thus len(symbol_sequence) equals len(transmitted_sequence).

    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2] 

    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]

    """
    s = [0.7071067811865475 + 0.7071067811865475j, 0.7071067811865475 + 0.7071067811865475j]
    symbols = [0.7071067811865475 + 0.7071067811865475j, -0.7071067811865475 + 0.7071067811865475j, -0.7071067811865475 - 0.7071067811865475j, 0.7071067811865475 - 0.7071067811865475j]
    c = impulse_response
    r = symbol_sequence
    h = 2
    x = 0

    while x < len(r):
        delta = []
        added = 0
        z = len(s) - 1
        w = 1

        while (z > -1) and (w < len(c)):
            added += (s[z] * c[w])
            w += 1
            z -= 1

        y = 0
        while y < len(symbols):
            dfe = abs(r[x] - (symbols[y] * c[0] + added)) ** 2
            delta.append(dfe)
            y += 1

        index = delta.index(min(delta))
        s.append(symbols[index])
        x += 1

    transmitted_sequence = s[h:]

    return transmitted_sequence

def MLSE_BPSK_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the MLSE algorithm on the given symbol sequence (which was modulated using the BPSK scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [1, 1]
    !!! NOTE: The appended symbols should be included in the given symbol sequence, thus if the block size is 200, then the length of the given symbol sequence should be 202.
    Only the transmitted data bits must be returned, thus exluding the prepended symbols AND the appended symbols. Thus is the block size is 200 then len(transmitted_sequence) should be 200.
    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]
    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """
    transmitted_sequence = []

    c0,c1,c2 = impulse_response

    row33 = []
    row31 = []
    row23 = []
    row21 = []
    row12 = []
    row10 = []
    row02 = []
    row00 = []

    pCost3 = [0]
    pCost2 = [0]
    pCost1 = [0]
    pCost0 = [0]

    time = 1
    r_t = symbol_sequence[(time - 1)]

    row33.append(pow(abs(r_t - (1 * c0 + 1 * c1 + 1 * c2)), 2))
    row31.append(pow(abs(r_t - (-1 * c0 + 1 * c1 + 1 * c2)), 2))
    row23.append(999)
    row21.append(999)
    row12.append(999)
    row10.append(999)
    row02.append(999)
    row00.append(999)

    if pCost3[time - 1] + row33[time - 1] < pCost2[time - 1] + row23[time - 1]:
        pCost3.append(pCost3[time - 1] + row33[time - 1])
    else:
        pCost3.append(pCost2[time - 1] + row23[time - 1])
    if pCost1[time - 1] + row12[time - 1] < pCost0[time - 1] + row02[time - 1]:
        pCost2.append(pCost1[time - 1] + row12[time - 1])
    else:
        pCost2.append(pCost0[time - 1] + row02[time - 1])
    if pCost3[time - 1] + row31[time - 1] < pCost2[time - 1] + row21[time - 1]:
        pCost1.append(pCost3[time - 1] + row31[time - 1])
    else:
        pCost1.append(pCost2[time - 1] + row21[time - 1])
    if pCost1[time - 1] + row10[time - 1] < pCost0[time - 1] + row00[time - 1]:
        pCost0.append(pCost1[time - 1] + row10[time - 1])
    else:
        pCost0.append(pCost0[time - 1] + row00[time - 1])

    time = 2
    r_t = symbol_sequence[(time - 1)]

    row33.append(pow(abs(r_t - (1 * c0 + 1 * c1 + 1 * c2)), 2))
    row31.append(pow(abs(r_t - (-1 * c0 + 1 * c1 + 1 * c2)), 2))
    row23.append(999)
    row21.append(999)
    row12.append(pow(abs(r_t - (1 * c0 + -1 * c1 + 1 * c2)), 2))
    row10.append(pow(abs(r_t - (-1 * c0 + -1 * c1 + 1 * c2)), 2))
    row02.append(999)
    row00.append(999)

    #Update Path Costs
    if pCost3[time - 1] + row33[time - 1] < pCost2[time - 1] + row23[time - 1]:
        pCost3.append(pCost3[time - 1] + row33[time - 1])
    else:
        pCost3.append(pCost2[time - 1] + row23[time - 1])
    if pCost1[time - 1] + row12[time - 1] < pCost0[time - 1] + row02[time - 1]:
        pCost2.append(pCost1[time - 1] + row12[time - 1])
    else:
        pCost2.append(pCost0[time - 1] + row02[time - 1])
    if pCost3[time - 1] + row31[time - 1] < pCost2[time - 1] + row21[time - 1]:
        pCost1.append(pCost3[time - 1] + row31[time - 1])
    else:
        pCost1.append(pCost2[time - 1] + row21[time - 1])
    if pCost1[time - 1] + row10[time - 1] < pCost0[time - 1] + row00[time - 1]:
        pCost0.append(pCost1[time - 1] + row10[time - 1])
    else:
        pCost0.append(pCost0[time - 1] + row00[time - 1])

    time = 3

    while time < len(symbol_sequence) - 1:
        r_t = symbol_sequence[(time - 1)]

        row33.append(pow(abs(r_t - (1 * c0 + 1 * c1 + 1 * c2)), 2))
        row31.append(pow(abs(r_t - (-1 * c0 + 1 * c1 + 1 * c2)), 2))
        row23.append(pow(abs(r_t - (1 * c0 + 1 * c1 + -1 * c2)), 2))
        row21.append(pow(abs(r_t - (-1 * c0 + 1 * c1 + -1 * c2)), 2))
        row12.append(pow(abs(r_t - (1 * c0 + -1 * c1 + 1 * c2)), 2))
        row10.append(pow(abs(r_t - (-1 * c0 + -1 * c1 + 1 * c2)), 2))
        row02.append(pow(abs(r_t - (1 * c0 + -1 * c1 + -1 * c2)), 2))
        row00.append(pow(abs(r_t - (-1 * c0 + -1 * c1 + -1 * c2)), 2))

        if pCost3[time - 1] + row33[time - 1] < pCost2[time - 1] + row23[time - 1]:
            pCost3.append(pCost3[time - 1] + row33[time - 1])
        else:
            pCost3.append(pCost2[time - 1] + row23[time - 1])
        if pCost1[time - 1] + row12[time - 1] < pCost0[time - 1] + row02[time - 1]:
            pCost2.append(pCost1[time - 1] + row12[time - 1])
        else:
            pCost2.append(pCost0[time - 1] + row02[time - 1])
        if pCost3[time - 1] + row31[time - 1] < pCost2[time - 1] + row21[time - 1]:
            pCost1.append(pCost3[time - 1] + row31[time - 1])
        else:
            pCost1.append(pCost2[time - 1] + row21[time - 1])
        if pCost1[time - 1] + row10[time - 1] < pCost0[time - 1] + row00[time - 1]:
            pCost0.append(pCost1[time - 1] + row10[time - 1])
        else:
            pCost0.append(pCost0[time - 1] + row00[time - 1])

        time = time + 1

    time = len(symbol_sequence) - 1
    r_t = symbol_sequence[(time - 1)]

    row33.append(pow(abs(r_t - (1 * c0 + 1 * c1 + 1 * c2)), 2))
    row31.append(999)
    row23.append(pow(abs(r_t - (1 * c0 + 1 * c1 + -1 * c2)), 2))
    row21.append(999)
    row12.append(pow(abs(r_t - (1 * c0 + -1 * c1 + 1 * c2)), 2))
    row10.append(999)
    row02.append(pow(abs(r_t - (1 * c0 + -1 * c1 + -1 * c2)), 2))
    row00.append(999)

    if pCost3[time - 1] + row33[time - 1] < pCost2[time - 1] + row23[time - 1]:
        pCost3.append(pCost3[time - 1] + row33[time - 1])
    else:
        pCost3.append(pCost2[time - 1] + row23[time - 1])
    if pCost1[time - 1] + row12[time - 1] < pCost0[time - 1] + row02[time - 1]:
        pCost2.append(pCost1[time - 1] + row12[time - 1])
    else:
        pCost2.append(pCost0[time - 1] + row02[time - 1])
    if pCost3[time - 1] + row31[time - 1] < pCost2[time - 1] + row21[time - 1]:
        pCost1.append(pCost3[time - 1] + row31[time - 1])
    else:
        pCost1.append(pCost2[time - 1] + row21[time - 1])
    if pCost1[time - 1] + row10[time - 1] < pCost0[time - 1] + row00[time - 1]:
        pCost0.append(pCost1[time - 1] + row10[time - 1])
    else:
        pCost0.append(pCost0[time - 1] + row00[time - 1])

    time = len(symbol_sequence)
    r_t = symbol_sequence[(time - 1)]

    row33.append(pow(abs(r_t - (1 * c0 + 1 * c1 + 1 * c2)), 2))
    row31.append(999)
    row23.append(pow(abs(r_t - (1 * c0 + 1 * c1 + -1 * c2)), 2))
    row21.append(999)
    row12.append(999)
    row10.append(999)
    row02.append(999)
    row00.append(999)

    if pCost3[time - 1] + row33[time - 1] < pCost2[time - 1] + row23[time - 1]:
        pCost3.append(pCost3[time - 1] + row33[time - 1])
    else:
        pCost3.append(pCost2[time - 1] + row23[time - 1])
    if pCost1[time - 1] + row12[time - 1] < pCost0[time - 1] + row02[time - 1]:
        pCost2.append(pCost1[time - 1] + row12[time - 1])
    else:
        pCost2.append(pCost0[time - 1] + row02[time - 1])
    if pCost3[time - 1] + row31[time - 1] < pCost2[time - 1] + row21[time - 1]:
        pCost1.append(pCost3[time - 1] + row31[time - 1])
    else:
        pCost1.append(pCost2[time - 1] + row21[time - 1])
    if pCost1[time - 1] + row10[time - 1] < pCost0[time - 1] + row00[time - 1]:
        pCost0.append(pCost1[time - 1] + row10[time - 1])
    else:
        pCost0.append(pCost0[time - 1] + row00[time - 1])

    numStates = 3

    counter = len(pCost3) - 1

    while counter > 0:
        if numStates == 3:
            transmitted_sequence = [1] + transmitted_sequence

            if pCost3[counter - 1] + row33[counter - 1] < pCost2[counter - 1] + row23[counter - 1]:
                numStates = 3
            else:
                numStates = 2
        elif numStates == 2:
            transmitted_sequence = [1] + transmitted_sequence

            if pCost1[counter - 1] + row12[counter - 1] < pCost0[counter - 1] + row02[counter - 1]:
                numStates = 1
            else:
                numStates = 0
        elif numStates == 1:
            transmitted_sequence = [-1] + transmitted_sequence

            if pCost3[counter - 1] + row31[counter - 1] < pCost2[counter - 1] + row21[counter - 1]:
                numStates = 3
            else:
                numStates = 2
        elif numStates == 0:
            transmitted_sequence = [-1] + transmitted_sequence

            if pCost1[counter - 1] + row10[counter - 1] < pCost0[counter - 1] + row00[counter - 1]:
                numStates = 1
            else:
                numStates = 0

        counter = counter - 1

    transmitted_sequence = transmitted_sequence[:-2]

    return transmitted_sequence

def MLSE_4QAM_BLOCK(symbol_sequence: SYMBOL_SEQUENCE_TYPE, impulse_response: CHANNEL_IMPULSE_RESPONSE_TYPE) -> SYMBOL_SEQUENCE_TYPE:
    """
    Performs the MLSE algorithm on the given symbol sequence (which was modulated using the 4QAM scheme) with the given impulse response, and returns the most probable transmitted symbol sequence.
    The impulse response length can be assumed to be 3, and the prepended symbols for t=-1 and t=-2 can be assumed to be [(0.7071067811865475+0.7071067811865475j), (0.7071067811865475+0.7071067811865475j)]
    !!! NOTE: The appended symbols should be included in the given symbol sequence, thus if the block size is 200, then the length of the given symbol sequence should be 202.
    Only the transmitted data bits must be returned, thus exluding the prepended symbols AND the appended symbols. Thus is the block size is 200 then len(transmitted_sequence) should be 200.
    parameters:
        symbol_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
        impulse_response -> type <class 'list'> : List containing complex items (<class "complex">) which represents the impulse response coeficients for example [1+1j, 2+2j, -0.66-0.25j] represents [c0, c1, c2]
    returns:
        transmitted_sequence -> type <class 'list'> : List containing complex items (<class "complex">) which represents the symbols for example: [1+1j, 2+2j, -0.66-0.25j]
    """
    symbols = [0.7071067811865475 + 0.7071067811865475j, -0.7071067811865475 + 0.7071067811865475j, -0.7071067811865475 - 0.7071067811865475j, 0.7071067811865475 - 0.7071067811865475j]
    c0, c1, c2 = impulse_response
    blockSize = len(symbol_sequence) - 2

    # Initializing trellis and path cost storage
    trellis = [[0] * blockSize for _ in range(4)]  # 4 symbols
    pathCosts = [[float('inf')] * blockSize for _ in range(4)]
    prevState = [[0] * blockSize for _ in range(4)]

    # Initial conditions: state at t = -2 and t = -1 are the same symbols
    for state in range(4):
        trellis[state][0] = 0.7071067811865475 + 0.7071067811865475j
        pathCosts[state][0] = 0  # Start with 0 cost for the initial state

    # Viterbi algorithm
    for t in range(2, blockSize):
        received_symbol = symbol_sequence[t]

        for current_state in range(4):
            current_symbol = symbols[current_state]
            min_cost = float('inf')
            best_prev_state = 0

            for prev_state_idx in range(4):
                prev_symbol = symbols[prev_state_idx]
                predicted_symbol = c0 * current_symbol + c1 * prev_symbol + c2 * trellis[prev_state_idx][t - 2]
                cost = pathCosts[prev_state_idx][t - 1] + abs(received_symbol - predicted_symbol) ** 2

                if cost < min_cost:  # Find the minimum cost
                    min_cost = cost
                    best_prev_state = prev_state_idx

            # Update trellis & path costs
            trellis[current_state][t] = current_symbol
            pathCosts[current_state][t] = min_cost
            prevState[current_state][t] = best_prev_state

    # Find the state with the minimum path cost at the last time step, avoiding states with inf costs
    min_cost = float('inf')
    bestFinalState = 0

    for state in range(4):
        if pathCosts[state][-1] < min_cost:
            min_cost = pathCosts[state][-1]
            bestFinalState = state

    transmitted_sequence = [0] * blockSize
    transmitted_sequence[-1] = symbols[bestFinalState]

    # Backtrack to recover sequence
    for t in range(blockSize - 2, -1, -1):
        bestFinalState = prevState[bestFinalState][t + 1]
        transmitted_sequence[t] = symbols[bestFinalState]

    return transmitted_sequence[2:]

#########################################################################
#                         SER and BER Functions                         #
#########################################################################

# BPSK

def SER_BER_BPSK_DFE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add noise to the symbol sequence in each block using the equation in the practical guide and the static impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 1
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)
    c_static = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        noisySymbols = []
        xReal = random_values_for_symbols[i][0][0] #process first symbol
        noise = sigma * xReal
        st = symbol_blocks[i][0]
        stMinus1 = 1
        stMinus2 = 1
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0] #process the second symbol
        noise = sigma * xReal
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 1
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            noise = sigma * xReal
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        transmitted_sequence += DFE_BPSK_BLOCK(noisy_symbol_blocks[i], c_static)

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER += 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_BPSK)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_BPSK_DFE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence in each block using the equation in the practical guide and the dynamic impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 1
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        noisySymbols = []
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)

        xReal = random_values_for_symbols[i][0][0] #process first symbol
        noise = sigma * xReal
        st = symbol_blocks[i][0]
        stMinus1 = 1
        stMinus2 = 1
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0] #process the second symbol
        noise = sigma * xReal
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 1
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            noise = sigma * xReal
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)

        transmitted_sequence += DFE_BPSK_BLOCK(noisy_symbol_blocks[i], [c0, c1, c2])

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_BPSK)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_BPSK_MLSE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [1, 1]
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10

    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.

    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
            
    """

    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 1
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)
    c_static = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        noisySymbols = []

        xReal = random_values_for_symbols[i][0][0]  # process first symbol
        noise = sigma * xReal
        st = symbol_blocks[i][0]
        stMinus1 = 1
        stMinus2 = 1
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0]  # process the second symbol
        noise = sigma * xReal
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 1
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            noise = sigma * xReal
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        transmitted_sequence += MLSE_BPSK_BLOCK(noisy_symbol_blocks[i], c_static)

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_BPSK)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_BPSK_MLSE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [1, 1]
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            !!! NOTE: The imaginary component in this case will be 0 since for BPSK no imaginary noise will be added and the other equation shown in the pre-practical slides should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 1
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_BPSK)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        noisySymbols = []

        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)

        xReal = random_values_for_symbols[i][0][0]  # process first symbol
        noise = sigma * xReal
        st = symbol_blocks[i][0]
        stMinus1 = 1
        stMinus2 = 1
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0]  # process the second symbol
        noise = sigma * xReal
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 1
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            noise = sigma * xReal
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)

        transmitted_sequence += MLSE_BPSK_BLOCK(noisy_symbol_blocks[i], [c0, c1, c2])

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_BPSK)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

# 4QAM

def SER_BER_4QAM_DFE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - add noise to the symbol sequence in each block using the equation in the practical guide and the static impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 2
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)
    c_static = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        noisySymbols = []
        xReal = random_values_for_symbols[i][0][0]  # process first symbol
        xImg = random_values_for_symbols[i][0][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][0]
        stMinus1 = 0.7071067811865475 + 0.7071067811865475j
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0]  # process the second symbol
        xImg = random_values_for_symbols[i][1][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            xImg = random_values_for_symbols[i][j][1]
            noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        transmitted_sequence += DFE_4QAM_BLOCK(noisy_symbol_blocks[i], c_static)

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_4QAM)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_4QAM_DFE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence in each block using the equation in the practical guide and the dynamic impulse response and SNR
        - save the noisy symbol sequence from each block
        - perform the DFE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 2
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)

        noisySymbols = []
        xReal = random_values_for_symbols[i][0][0]  # process first symbol
        xImg = random_values_for_symbols[i][0][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][0]
        stMinus1 = 0.7071067811865475 + 0.7071067811865475j
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0]  # process the second symbol
        xImg = random_values_for_symbols[i][1][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            xImg = random_values_for_symbols[i][j][1]
            noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)
        transmitted_sequence += DFE_4QAM_BLOCK(noisy_symbol_blocks[i], [c0, c1, c2])

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_4QAM)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_4QAM_MLSE_STATIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using 4QAM
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For 4QAM the appended symbols are [0.7071067811865475+0.7071067811865475j, 0.7071067811865475+0.7071067811865475j]
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 2
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)
    c_static = [0.29 + 0.98j, 0.73 - 0.24j, 0.21 + 0.91j]

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        noisySymbols = []
        xReal = random_values_for_symbols[i][0][0]  # process first symbol
        xImg = random_values_for_symbols[i][0][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][0]
        stMinus1 = 0.7071067811865475 + 0.7071067811865475j
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0]  # process the second symbol
        xImg = random_values_for_symbols[i][1][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            xImg = random_values_for_symbols[i][j][1]
            noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c_static[0] + stMinus1 * c_static[1] + stMinus2 * c_static[2] + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        transmitted_sequence += MLSE_4QAM_BLOCK(noisy_symbol_blocks[i], c_static)

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_4QAM)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

def SER_BER_4QAM_MLSE_DYNAMIC(bit_sequence: BIT_SEQUENCE_TYPE, block_size: int, random_values_for_symbols: RANDOM_VALUES_SYMBOLS_TYPE, random_values_for_CIR: RANDOM_VALUES_CIR_TYPE, snr: float) -> Tuple[SYMBOL_BLOCKS_TYPE, SER_TYPE, BER_TYPE]:
    """
    This function uses the given bit sequence, block size, SNR value and random values to perform a simulation that:
        - converts the bit sequence into a symbol sequence using BPSK
        - splits the symbol sequence into blocks with the given block size
        - add the appended symbols to each sequence in each block. For BPSK the appended symbols are [0.7071067811865475+0.7071067811865475j, 0.7071067811865475+0.7071067811865475j]
        - calculate a channel impulse response for each block using the dynamic impulse response equation in the practical guide
        - add noise to the symbol sequence including the appended symbols in each block using the equation in the practical guide and the static impulse response and SNR.
        - save the noisy symbol sequence from each block which should all be block size + 2 long, thus including the symbols that was appended.
        - perform the MLSE algorithm on each noisy symbol sequence in the blocks to get the corresponding transmitted sequences excluding the appended symbols.
        - combine the blocks of transmitted sequences into a single symbol sequence
        - calculate the SER and BER by comparing the given sequences and transmitted sequences
        - return the blocks of noisy symbols, the SER value scale to log10, and the BER value scale to log10
    parameters:
        bit_sequence -> type <class 'list'> : List containing int items which represents the bits for example: [0, 1, 1, 1, 0, 0, 1, 1]  
        block_size -> type <class 'int'> : An integer indicating the amount of symbols that should be within each block
        random_values_for_symbols -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_symbols[0][0][0] and returns a float.
            These random values are used to calculate the added noise according to equation 1 given in the practical guide.
            The first index indicates the corresponding block. The second index indicate the corresponding symbol in that block. The third index indicate the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_symbols[5][84][0] -> is the float that should be used for the real component kappa value to calculate the noise for the 85th symbol in the 6th block (note: the first block is index 0).
        random_values_for_CIR -> type <class 'list'> : A list containing lists containing lists. Thus the variable has three index values random_values_for_CIR[0][0][0] and returns a float.
            These random values are used to calculate the dynamic impulse response using equation 4 in the practical guide
            The first index indicates the corresponding block. The second index indicate the corresponding coeficient (c0 - [0], c1 = [1], c2 - [2]). The third index indicates the real (0) or imaginary (1) component for which the float value should be used.
            Thus -> random_values_for_CIR[7][2][1] -> is the float that should be used for the imaginary component kappa value to calculate the c2 coeficient for the 8th block (note: the first block is index 0).
        snr -> type <class 'float'> : A float value giving the SNR value that should be used for calculating the sigma value used in the noise equation.
    returns:
        noisy_symbol_blocks -> type <class 'list'> : List containing lists. Each list entry contains a sequence of <class "complex"> which represents the symbols after noise was added for each block
          Example, blocks of size 3 :
            [
                [(-0.262+0.757j), ( 0.071+0.161j), (0.255-0.333j)],
                [( 0.247-0.471j), (-0.668-0.256j), (0.032-0.345j)],
                [(-0.091+0.641j), ( 0.535+0.101j), (0.203+0.614j)],
                [(-0.256-0.217j), ( 0.577-0.327j), (0.347-0.359j)],
            ]
        SER -> type <class 'float'> or <class 'NoneType'> : Returns the SER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero. 
        BER -> type <class 'float'> or <class 'NoneType'> : Returns the BER value (not scaled) or returns the NoneType (return None) if the amount of errors is zero.
    """
    noisy_symbol_blocks = []
    SER = 0
    BER = 0

    fBit = 2
    sigma = 1 / cm.sqrt(pow(10, 0.1 * snr) * fBit)

    symbol_sequence = assist_bit_to_symbol(bit_sequence, MAP_4QAM)
    symbol_blocks = assist_split_symbols_into_blocks(symbol_sequence, block_size)

    for i in range(len(symbol_blocks)):
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)

        noisySymbols = []
        xReal = random_values_for_symbols[i][0][0]  # process first symbol
        xImg = random_values_for_symbols[i][0][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][0]
        stMinus1 = 0.7071067811865475 + 0.7071067811865475j
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        xReal = random_values_for_symbols[i][1][0]  # process the second symbol
        xImg = random_values_for_symbols[i][1][1]

        noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
        st = symbol_blocks[i][1]
        stMinus1 = symbol_blocks[i][0]
        stMinus2 = 0.7071067811865475 + 0.7071067811865475j
        rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
        noisySymbols.append(rt)

        for j in range(2, len(symbol_blocks[i])):  # Rest of the symbols
            xReal = random_values_for_symbols[i][j][0]
            xImg = random_values_for_symbols[i][j][1]
            noise = sigma * ((xReal + xImg * 1j) / cm.sqrt(2))
            st = symbol_blocks[i][j]
            stMinus1 = symbol_blocks[i][j - 1]
            stMinus2 = symbol_blocks[i][j - 2]
            rt = st * c0 + stMinus1 * c1 + stMinus2 * c2 + noise
            noisySymbols.append(rt)

        noisy_symbol_blocks.append(noisySymbols)

    transmitted_sequence = []
    for i in range(len(noisy_symbol_blocks)):
        c0 = (random_values_for_CIR[i][0][0] + random_values_for_CIR[i][0][1] * 1j) / cm.sqrt(2 * 3)
        c1 = (random_values_for_CIR[i][1][0] + random_values_for_CIR[i][1][1] * 1j) / cm.sqrt(2 * 3)
        c2 = (random_values_for_CIR[i][2][0] + random_values_for_CIR[i][2][1] * 1j) / cm.sqrt(2 * 3)
        transmitted_sequence += MLSE_4QAM_BLOCK(noisy_symbol_blocks[i], [c0, c1, c2])

    for i in range(len(transmitted_sequence)):
        if transmitted_sequence[i] != symbol_sequence[i]:
            SER = SER + 1

    if SER != 0:
        SER = SER / len(transmitted_sequence)
    else:
        SER = None

    transmittedBitSequence = assist_symbol_to_bit(transmitted_sequence, MAP_4QAM)

    for i in range(len(transmittedBitSequence)):
        if transmittedBitSequence[i] != bit_sequence[i]:
            BER = BER + 1

    if BER != 0:
        BER = BER / len(transmittedBitSequence)
    else:
        BER = None

    return noisy_symbol_blocks, SER, BER

####### DO NOT EDIT #######
if __name__ == "__main__" :

    evaluate()
####### DO NOT EDIT #######